#import <Foundation/Foundation.h>
#import <Virtualization/Virtualization.h>

@interface VMDelegate : NSObject <VZVirtualMachineDelegate>
@property(nonatomic, copy) void (^stopHandler)(NSString *);
@end

@implementation VMDelegate
- (void)guestDidStopVirtualMachine:(VZVirtualMachine *)virtualMachine
{
    if (self.stopHandler) {
        self.stopHandler(@"guest stopped");
    }
}

- (void)virtualMachine:(VZVirtualMachine *)virtualMachine didStopWithError:(NSError *)error
{
    if (self.stopHandler) {
        self.stopHandler([NSString stringWithFormat:@"guest stopped with error: %@", error]);
    }
}
@end

static uint64_t monotonicNanos(void)
{
    return dispatch_time(DISPATCH_TIME_NOW, 0);
}

int main(int argc, const char *argv[])
{
    @autoreleasepool {
        if (argc < 4) {
            fprintf(stderr, "usage: minivz_objc <kernel> <initrd> <serial.log> [timeout-seconds]\n");
            return 64;
        }

        NSString *kernelPath = [NSString stringWithUTF8String:argv[1]];
        NSString *initrdPath = [NSString stringWithUTF8String:argv[2]];
        NSString *serialLogPath = [NSString stringWithUTF8String:argv[3]];
        NSTimeInterval timeout = argc >= 5 ? [[NSString stringWithUTF8String:argv[4]] doubleValue] : 30.0;

        NSURL *kernelURL = [NSURL fileURLWithPath:kernelPath];
        NSURL *initrdURL = [NSURL fileURLWithPath:initrdPath];

        VZLinuxBootLoader *bootLoader = [[VZLinuxBootLoader alloc] initWithKernelURL:kernelURL];
        bootLoader.initialRamdiskURL = initrdURL;
        bootLoader.commandLine = @"console=hvc0 rdinit=/bin/sh printk.devkmsg=on";

        NSPipe *consoleInput = [NSPipe pipe];
        NSPipe *consoleOutput = [NSPipe pipe];
        VZFileHandleSerialPortAttachment *attachment =
            [[VZFileHandleSerialPortAttachment alloc] initWithFileHandleForReading:consoleInput.fileHandleForReading
                                                             fileHandleForWriting:consoleOutput.fileHandleForWriting];

        VZVirtioConsoleDeviceSerialPortConfiguration *serialPort =
            [[VZVirtioConsoleDeviceSerialPortConfiguration alloc] init];
        serialPort.attachment = attachment;

        VZGenericPlatformConfiguration *platform = [[VZGenericPlatformConfiguration alloc] init];
        platform.machineIdentifier = [[VZGenericMachineIdentifier alloc] init];

        VZVirtualMachineConfiguration *configuration = [[VZVirtualMachineConfiguration alloc] init];
        configuration.bootLoader = bootLoader;
        configuration.CPUCount = 2;
        configuration.memorySize = 512ull * 1024ull * 1024ull;
        configuration.platform = platform;
        configuration.serialPorts = @[ serialPort ];
        configuration.entropyDevices = @[ [[VZVirtioEntropyDeviceConfiguration alloc] init] ];

        NSError *validationError = nil;
        if (![configuration validateWithError:&validationError]) {
            fprintf(stderr, "configuration invalid: %s\n", validationError.localizedDescription.UTF8String);
            return 1;
        }

        [[NSFileManager defaultManager] createFileAtPath:serialLogPath contents:nil attributes:nil];
        NSFileHandle *serialLogHandle = [NSFileHandle fileHandleForWritingAtPath:serialLogPath];
        if (serialLogHandle == nil) {
            fprintf(stderr, "failed to open serial log: %s\n", serialLogPath.UTF8String);
            return 1;
        }

        __block BOOL sawFirstOutput = NO;
        uint64_t startedAt = monotonicNanos();
        consoleOutput.fileHandleForReading.readabilityHandler = ^(NSFileHandle *handle) {
            NSData *data = handle.availableData;
            if (data.length == 0) {
                return;
            }

            if (!sawFirstOutput) {
                sawFirstOutput = YES;
                NSTimeInterval seconds = (double)(monotonicNanos() - startedAt) / (double)NSEC_PER_SEC;
                fprintf(stderr, "\n[VZ] first console output: %.3fs\n", seconds);
            }

            [serialLogHandle writeData:data];
            write(STDOUT_FILENO, data.bytes, data.length);
        };

        VZVirtualMachine *vm = [[VZVirtualMachine alloc] initWithConfiguration:configuration];

        dispatch_semaphore_t stopSemaphore = dispatch_semaphore_create(0);
        VMDelegate *delegate = [[VMDelegate alloc] init];
        delegate.stopHandler = ^(NSString *message) {
            fprintf(stderr, "\n[VZ] %s\n", message.UTF8String);
            dispatch_semaphore_signal(stopSemaphore);
        };
        vm.delegate = delegate;

        __block NSError *startError = nil;
        __block BOOL started = NO;
        [vm startWithCompletionHandler:^(NSError * _Nullable errorOrNil) {
            startError = errorOrNil;
            started = (errorOrNil == nil);
            dispatch_semaphore_signal(stopSemaphore);
        }];

        while (!started && startError == nil) {
            dispatch_semaphore_wait(stopSemaphore, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.1 * NSEC_PER_SEC)));
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
        }

        if (startError != nil) {
            fprintf(stderr, "failed to start VM: %s\n", startError.localizedDescription.UTF8String);
            return 1;
        }

        if (!started) {
            fprintf(stderr, "VM did not start and no error was reported\n");
            return 1;
        }

        fprintf(stderr, "[VZ] VM started\n");

        NSString *guestCommand = @"echo __VZ_PROBE_READY__\nuname -a\ncat /proc/meminfo | head -n 3\necho __VZ_DONE__\n";
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(4 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
            NSData *commandData = [guestCommand dataUsingEncoding:NSUTF8StringEncoding];
            [consoleInput.fileHandleForWriting writeData:commandData];
        });

        NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
        while ([deadline timeIntervalSinceNow] > 0) {
            if (dispatch_semaphore_wait(stopSemaphore, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.25 * NSEC_PER_SEC))) == 0) {
                break;
            }
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
        }

        if (!sawFirstOutput) {
            fprintf(stderr, "timed out after %.1fs waiting for guest output\n", timeout);
            return 2;
        }

        return 0;
    }
}
