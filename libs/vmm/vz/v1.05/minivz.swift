import Dispatch
import Foundation
import Virtualization

enum ProbeError: Error, CustomStringConvertible {
    case usage(String)
    case timeout(seconds: TimeInterval)

    var description: String {
        switch self {
        case .usage(let message):
            return message
        case .timeout(let seconds):
            return "timed out after \(seconds)s waiting for guest output"
        }
    }
}

final class ConsoleSink {
    private let readHandle: FileHandle
    private let logHandle: FileHandle
    private let startedAt: DispatchTime
    private var buffer = Data()
    private var firstOutputRecorded = false
    private let firstOutputHook: (TimeInterval) -> Void

    init(
        readHandle: FileHandle,
        logURL: URL,
        startedAt: DispatchTime,
        firstOutputHook: @escaping (TimeInterval) -> Void
    ) throws {
        self.readHandle = readHandle
        self.startedAt = startedAt
        self.firstOutputHook = firstOutputHook
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        self.logHandle = try FileHandle(forWritingTo: logURL)
        self.readHandle.readabilityHandler = { [weak self] handle in
            self?.drain(handle: handle)
        }
    }

    deinit {
        readHandle.readabilityHandler = nil
        try? logHandle.close()
    }

    private func drain(handle: FileHandle) {
        let data = handle.availableData
        guard !data.isEmpty else {
            return
        }

        if !firstOutputRecorded {
            firstOutputRecorded = true
            let deltaNs = DispatchTime.now().uptimeNanoseconds - startedAt.uptimeNanoseconds
            firstOutputHook(TimeInterval(deltaNs) / 1_000_000_000)
        }

        buffer.append(data)
        try? logHandle.write(contentsOf: data)
        if let text = String(data: data, encoding: .utf8) {
            FileHandle.standardOutput.write(Data(text.utf8))
        } else {
            let rendered = data.map { String(format: "%02x", $0) }.joined(separator: " ")
            FileHandle.standardOutput.write(Data("[serial hex] \(rendered)\n".utf8))
        }
    }
}

final class VMDelegate: NSObject, VZVirtualMachineDelegate {
    let stopHandler: (String) -> Void

    init(stopHandler: @escaping (String) -> Void) {
        self.stopHandler = stopHandler
    }

    func guestDidStop(_ virtualMachine: VZVirtualMachine) {
        stopHandler("guest stopped")
    }

    func virtualMachine(_ virtualMachine: VZVirtualMachine, didStopWithError error: Error) {
        stopHandler("guest stopped with error: \(error)")
    }
}

@main
struct MiniVzProbe {
    static func main() async {
        do {
            try await run()
            exit(EXIT_SUCCESS)
        } catch {
            fputs("minivz: \(error)\n", stderr)
            exit(EXIT_FAILURE)
        }
    }

    private static func run() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count >= 4 else {
            throw ProbeError.usage(
                "usage: minivz <kernel> <initrd> <serial.log> [timeout-seconds]"
            )
        }

        let kernelURL = URL(fileURLWithPath: arguments[1])
        let initrdURL = URL(fileURLWithPath: arguments[2])
        let serialLogURL = URL(fileURLWithPath: arguments[3])
        let timeout = arguments.count >= 5 ? (TimeInterval(arguments[4]) ?? 30) : 30

        let bootLoader = VZLinuxBootLoader(kernelURL: kernelURL)
        bootLoader.initialRamdiskURL = initrdURL
        bootLoader.commandLine = "console=hvc0 rdinit=/bin/sh printk.devkmsg=on"

        let consoleInput = Pipe()
        let consoleOutput = Pipe()
        let serialAttachment = try VZFileHandleSerialPortAttachment(
            fileHandleForReading: consoleInput.fileHandleForReading,
            fileHandleForWriting: consoleOutput.fileHandleForWriting
        )

        let serialPort = VZVirtioConsoleDeviceSerialPortConfiguration()
        serialPort.attachment = serialAttachment

        let platform = VZGenericPlatformConfiguration()
        platform.machineIdentifier = VZGenericMachineIdentifier()

        let config = VZVirtualMachineConfiguration()
        config.bootLoader = bootLoader
        config.cpuCount = 2
        config.memorySize = 512 * 1024 * 1024
        config.platform = platform
        config.serialPorts = [serialPort]
        config.entropyDevices = [VZVirtioEntropyDeviceConfiguration()]
        config.networkDevices = []
        try config.validate()

        let startedAt = DispatchTime.now()
        var firstOutputSeconds: TimeInterval?
        let consoleSink = try ConsoleSink(
            readHandle: consoleOutput.fileHandleForReading,
            logURL: serialLogURL,
            startedAt: startedAt
        ) { seconds in
            firstOutputSeconds = seconds
            fputs(String(format: "\n[VZ] first console output: %.3fs\n", seconds), stderr)
        }
        _ = consoleSink

        let stopSemaphore = DispatchSemaphore(value: 0)
        let delegate = VMDelegate { reason in
            fputs("\n[VZ] \(reason)\n", stderr)
            stopSemaphore.signal()
        }

        let vm = VZVirtualMachine(configuration: config)
        vm.delegate = delegate

        try await withCheckedThrowingContinuation { continuation in
            vm.start { result in
                switch result {
                case .success:
                    continuation.resume()
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
        }

        fputs("[VZ] VM started\n", stderr)

        let guestCommand = """
        echo __VZ_PROBE_READY__
        uname -a
        cat /proc/meminfo | head -n 3
        echo __VZ_DONE__
        """
        DispatchQueue.global().asyncAfter(deadline: .now() + 4) {
            if let data = "\(guestCommand)\n".data(using: .utf8) {
                try? consoleInput.fileHandleForWriting.write(contentsOf: data)
            }
        }

        let deadline = DispatchTime.now() + timeout
        while DispatchTime.now() < deadline {
            if let output = firstOutputSeconds, output > 0, stopSemaphore.wait(timeout: .now()) == .success {
                break
            }
            if stopSemaphore.wait(timeout: .now() + .milliseconds(250)) == .success {
                break
            }
        }

        if stopSemaphore.wait(timeout: .now()) == .timedOut {
            if let data = "echo __VZ_TIMEOUT_EXIT__\n".data(using: .utf8) {
                try? consoleInput.fileHandleForWriting.write(contentsOf: data)
            }
        }

        if firstOutputSeconds == nil {
            throw ProbeError.timeout(seconds: timeout)
        }
    }
}
