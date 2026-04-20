#import <Foundation/Foundation.h>
#import <Virtualization/Virtualization.h>

#include <dispatch/dispatch.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static void print_usage(FILE *stream) {
    fprintf(stream,
            "usage: vz-vsock-runner --disk <path> --unix-socket <path> "
            "[--seed-disk <path>] [--nvram <path>] [--machine-id <path>] "
            "[--serial-log <path> | --stdio-console] [--nat-mac <mac>] "
            "[--vsock-port <port>] [--memory-mib <mib>] [--cpu-count <count>] "
            "[--enable-nat]\n");
}

static int connect_unix_socket(NSString *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket(AF_UNIX)");
        return -1;
    }

    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    const char *cpath = [path fileSystemRepresentation];
    if (strlen(cpath) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "unix socket path too long: %s\n", cpath);
        close(fd);
        return -1;
    }
    strncpy(addr.sun_path, cpath, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("connect(AF_UNIX)");
        close(fd);
        return -1;
    }
    return fd;
}

@class VzVsockRunner;

@interface BridgeSession : NSObject
@property(nonatomic, strong) VZVirtioSocketConnection *connection;
@property(nonatomic, assign) int unixFD;
@property(nonatomic, weak) VzVsockRunner *runner;
@property(nonatomic, assign) BOOL closed;
- (instancetype)initWithConnection:(VZVirtioSocketConnection *)connection
                            unixFD:(int)unixFD
                            runner:(VzVsockRunner *)runner;
- (void)start;
- (void)closeSession;
@end

@interface SocketListenerDelegate : NSObject <VZVirtioSocketListenerDelegate>
@property(nonatomic, weak) VzVsockRunner *runner;
- (instancetype)initWithRunner:(VzVsockRunner *)runner;
@end

@interface VmDelegate : NSObject <VZVirtualMachineDelegate>
@end

@interface VzVsockRunner : NSObject
@property(nonatomic, strong) VZVirtualMachine *vm;
@property(nonatomic, strong) SocketListenerDelegate *socketDelegate;
@property(nonatomic, strong) VmDelegate *vmDelegate;
@property(nonatomic, strong) NSMutableSet<BridgeSession *> *sessions;
@property(nonatomic, copy) NSString *unixSocketPath;
@property(nonatomic, assign) uint32_t vsockPort;
@property(nonatomic, strong) dispatch_queue_t vmQueue;
- (instancetype)initWithUnixSocketPath:(NSString *)unixSocketPath vsockPort:(uint32_t)vsockPort;
- (BOOL)startWithDiskPath:(NSString *)diskPath
               seedDiskPath:(nullable NSString *)seedDiskPath
                nvramPath:(nullable NSString *)nvramPath
            machineIDPath:(nullable NSString *)machineIDPath
               serialPath:(nullable NSString *)serialPath
         useStdioConsole:(BOOL)useStdioConsole
               memoryMiB:(NSUInteger)memoryMiB
                cpuCount:(NSUInteger)cpuCount
                  natMAC:(nullable NSString *)natMAC
               enableNAT:(BOOL)enableNAT
                   error:(NSError **)error;
- (void)addSession:(BridgeSession *)session;
- (void)removeSession:(BridgeSession *)session;
@end

@implementation BridgeSession

- (instancetype)initWithConnection:(VZVirtioSocketConnection *)connection
                            unixFD:(int)unixFD
                            runner:(VzVsockRunner *)runner {
    self = [super init];
    if (self) {
        _connection = connection;
        _unixFD = unixFD;
        _runner = runner;
        _closed = NO;
    }
    return self;
}

- (void)startPumpFromFD:(int)srcFD toFD:(int)dstFD {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        uint8_t buffer[8192];
        while (1) {
            ssize_t n = read(srcFD, buffer, sizeof(buffer));
            if (n == 0) {
                break;
            }
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }

            ssize_t written = 0;
            while (written < n) {
                ssize_t w = write(dstFD, buffer + written, (size_t)(n - written));
                if (w < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    [self closeSession];
                    return;
                }
                written += w;
            }
        }
        [self closeSession];
    });
}

- (void)start {
    int guestFD = self.connection.fileDescriptor;
    [self startPumpFromFD:guestFD toFD:self.unixFD];
    [self startPumpFromFD:self.unixFD toFD:guestFD];
}

- (void)closeSession {
    @synchronized (self) {
        if (self.closed) {
            return;
        }
        self.closed = YES;
        [self.connection close];
        if (self.unixFD >= 0) {
            close(self.unixFD);
            self.unixFD = -1;
        }
        [self.runner removeSession:self];
    }
}

@end

@implementation SocketListenerDelegate

- (instancetype)initWithRunner:(VzVsockRunner *)runner {
    self = [super init];
    if (self) {
        _runner = runner;
    }
    return self;
}

- (BOOL)listener:(VZVirtioSocketListener *)listener
shouldAcceptNewConnection:(VZVirtioSocketConnection *)connection
 fromSocketDevice:(VZVirtioSocketDevice *)socketDevice {
    (void)listener;
    (void)socketDevice;

    int unixFD = connect_unix_socket(self.runner.unixSocketPath);
    if (unixFD < 0) {
        fprintf(stderr, "failed to connect host unix socket: %s\n", self.runner.unixSocketPath.UTF8String);
        return NO;
    }

    BridgeSession *session = [[BridgeSession alloc] initWithConnection:connection unixFD:unixFD runner:self.runner];
    [self.runner addSession:session];
    [session start];
    fprintf(stderr, "accepted guest virtio-socket connection on port %u -> %s\n",
            connection.destinationPort, self.runner.unixSocketPath.UTF8String);
    return YES;
}

@end

@implementation VmDelegate

- (void)guestDidStopVirtualMachine:(VZVirtualMachine *)virtualMachine {
    (void)virtualMachine;
    fprintf(stderr, "guest stopped\n");
    exit(0);
}

- (void)virtualMachine:(VZVirtualMachine *)virtualMachine didStopWithError:(NSError *)error {
    (void)virtualMachine;
    fprintf(stderr, "guest stopped with error: %s\n", error.localizedDescription.UTF8String);
    exit(1);
}

@end

@implementation VzVsockRunner

- (instancetype)initWithUnixSocketPath:(NSString *)unixSocketPath vsockPort:(uint32_t)vsockPort {
    self = [super init];
    if (self) {
        _unixSocketPath = [unixSocketPath copy];
        _vsockPort = vsockPort;
        _sessions = [NSMutableSet set];
        _vmQueue = dispatch_queue_create("motlie.vfs.v1_15.vz_runner", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (void)addSession:(BridgeSession *)session {
    @synchronized (self.sessions) {
        [self.sessions addObject:session];
    }
}

- (void)removeSession:(BridgeSession *)session {
    @synchronized (self.sessions) {
        [self.sessions removeObject:session];
    }
}

- (BOOL)startWithDiskPath:(NSString *)diskPath
               seedDiskPath:(nullable NSString *)seedDiskPath
                nvramPath:(nullable NSString *)nvramPath
            machineIDPath:(nullable NSString *)machineIDPath
               serialPath:(nullable NSString *)serialPath
         useStdioConsole:(BOOL)useStdioConsole
               memoryMiB:(NSUInteger)memoryMiB
                cpuCount:(NSUInteger)cpuCount
                  natMAC:(nullable NSString *)natMAC
               enableNAT:(BOOL)enableNAT
                   error:(NSError **)error {
    NSURL *diskURL = [NSURL fileURLWithPath:diskPath];
    VZDiskImageStorageDeviceAttachment *attachment =
        [[VZDiskImageStorageDeviceAttachment alloc] initWithURL:diskURL
                                                       readOnly:NO
                                                    cachingMode:VZDiskImageCachingModeAutomatic
                                            synchronizationMode:VZDiskImageSynchronizationModeFsync
                                                          error:error];
    if (!attachment) {
        return NO;
    }

    VZVirtioBlockDeviceConfiguration *block = [[VZVirtioBlockDeviceConfiguration alloc] initWithAttachment:attachment];
    VZVirtioSocketDeviceConfiguration *socketConfig = [[VZVirtioSocketDeviceConfiguration alloc] init];
    NSMutableArray<VZStorageDeviceConfiguration *> *storageDevices = [NSMutableArray arrayWithObject:block];
    NSString *loggedNatMAC = @"(disabled)";

    NSMutableArray<VZUSBControllerConfiguration *> *usbControllers = [NSMutableArray array];
    if (seedDiskPath.length > 0) {
        NSURL *seedURL = [NSURL fileURLWithPath:seedDiskPath];
        VZDiskImageStorageDeviceAttachment *seedAttachment =
            [[VZDiskImageStorageDeviceAttachment alloc] initWithURL:seedURL
                                                           readOnly:YES
                                                        cachingMode:VZDiskImageCachingModeAutomatic
                                                synchronizationMode:VZDiskImageSynchronizationModeFsync
                                                              error:error];
        if (!seedAttachment) {
            return NO;
        }
        VZUSBMassStorageDeviceConfiguration *seedUSB =
            [[VZUSBMassStorageDeviceConfiguration alloc] initWithAttachment:seedAttachment];
        VZXHCIControllerConfiguration *xhci = [[VZXHCIControllerConfiguration alloc] init];
        xhci.usbDevices = @[ seedUSB ];
        [usbControllers addObject:xhci];
    }

    VZVirtualMachineConfiguration *config = [[VZVirtualMachineConfiguration alloc] init];
    config.CPUCount = cpuCount;
    config.memorySize = (uint64_t)memoryMiB * 1024ULL * 1024ULL;
    config.bootLoader = [[VZEFIBootLoader alloc] init];

    if (nvramPath.length > 0) {
        NSURL *nvramURL = [NSURL fileURLWithPath:nvramPath];
        if ([[NSFileManager defaultManager] fileExistsAtPath:nvramPath]) {
            ((VZEFIBootLoader *)config.bootLoader).variableStore = [[VZEFIVariableStore alloc] initWithURL:nvramURL];
        } else {
            VZEFIVariableStore *store = [[VZEFIVariableStore alloc] initCreatingVariableStoreAtURL:nvramURL
                                                                                           options:VZEFIVariableStoreInitializationOptionAllowOverwrite
                                                                                             error:error];
            if (!store) {
                return NO;
            }
            ((VZEFIBootLoader *)config.bootLoader).variableStore = store;
        }
    }

    VZGenericPlatformConfiguration *platform = [[VZGenericPlatformConfiguration alloc] init];
    if (machineIDPath.length > 0) {
        NSData *data = [NSData dataWithContentsOfFile:machineIDPath];
        if (data) {
            VZGenericMachineIdentifier *machineID = [[VZGenericMachineIdentifier alloc] initWithDataRepresentation:data];
            if (machineID) {
                platform.machineIdentifier = machineID;
            }
        }
        if (!platform.machineIdentifier) {
            VZGenericMachineIdentifier *machineID = [[VZGenericMachineIdentifier alloc] init];
            platform.machineIdentifier = machineID;
            [machineID.dataRepresentation writeToFile:machineIDPath atomically:YES];
        }
    }
    config.platform = platform;
    config.storageDevices = storageDevices;
    config.socketDevices = @[ socketConfig ];
    config.usbControllers = usbControllers;

    if (useStdioConsole || serialPath.length > 0) {
        VZVirtioConsoleDeviceSerialPortConfiguration *serial = [[VZVirtioConsoleDeviceSerialPortConfiguration alloc] init];
        if (useStdioConsole) {
            NSFileHandle *stdinHandle = [[NSFileHandle alloc] initWithFileDescriptor:STDIN_FILENO closeOnDealloc:NO];
            NSFileHandle *stdoutHandle = [[NSFileHandle alloc] initWithFileDescriptor:STDOUT_FILENO closeOnDealloc:NO];
            serial.attachment = [[VZFileHandleSerialPortAttachment alloc] initWithFileHandleForReading:stdinHandle
                                                                                   fileHandleForWriting:stdoutHandle];
        } else {
            NSError *serialError = nil;
            NSURL *serialURL = [NSURL fileURLWithPath:serialPath];
            VZFileSerialPortAttachment *serialAttachment = [[VZFileSerialPortAttachment alloc] initWithURL:serialURL append:YES error:&serialError];
            if (!serialAttachment) {
                if (error) {
                    *error = serialError;
                }
                return NO;
            }
            serial.attachment = serialAttachment;
        }
        config.serialPorts = @[ serial ];
    }

    if (enableNAT) {
        VZVirtioNetworkDeviceConfiguration *net = [[VZVirtioNetworkDeviceConfiguration alloc] init];
        net.attachment = [[VZNATNetworkDeviceAttachment alloc] init];
        VZMACAddress *macAddress = nil;
        if (natMAC.length > 0) {
            macAddress = [[VZMACAddress alloc] initWithString:natMAC];
            if (!macAddress) {
                if (error) {
                    NSString *message = [NSString stringWithFormat:@"invalid NAT MAC address: %@", natMAC];
                    *error = [NSError errorWithDomain:@"motlie.vfs.v1_15.vz"
                                                 code:3
                                             userInfo:@{NSLocalizedDescriptionKey: message}];
                }
                return NO;
            }
        } else {
            macAddress = [VZMACAddress randomLocallyAdministeredAddress];
        }
        net.MACAddress = macAddress;
        loggedNatMAC = macAddress.string;
        config.networkDevices = @[ net ];
    }

    if (![config validateWithError:error]) {
        return NO;
    }

    __block BOOL started = NO;
    __block NSError *startError = nil;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    dispatch_sync(self.vmQueue, ^{
        self.vm = [[VZVirtualMachine alloc] initWithConfiguration:config queue:self.vmQueue];
        self.vmDelegate = [[VmDelegate alloc] init];
        self.vm.delegate = self.vmDelegate;

        VZVirtioSocketDevice *socketDevice = (VZVirtioSocketDevice *)self.vm.socketDevices.firstObject;
        self.socketDelegate = [[SocketListenerDelegate alloc] initWithRunner:self];
        VZVirtioSocketListener *listener = [[VZVirtioSocketListener alloc] init];
        listener.delegate = self.socketDelegate;
        [socketDevice setSocketListener:listener forPort:self.vsockPort];

        [self.vm startWithCompletionHandler:^(NSError * _Nullable errorOrNil) {
            startError = errorOrNil;
            started = (errorOrNil == nil);
            dispatch_semaphore_signal(sem);
        }];
    });
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

    if (!started) {
        if (error) {
            *error = startError;
        }
        return NO;
    }

    fprintf(stderr, "vz-vsock-runner started: disk=%s seed=%s port=%u socket=%s nat-mac=%s\n",
            diskPath.UTF8String,
            seedDiskPath.length > 0 ? seedDiskPath.UTF8String : "(none)",
            self.vsockPort,
            self.unixSocketPath.UTF8String,
            loggedNatMAC.UTF8String);
    return YES;
}

@end

static void install_signal_handler(int signum, dispatch_block_t block) {
    signal(signum, SIG_IGN);
    dispatch_source_t source = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, signum, 0, dispatch_get_main_queue());
    dispatch_source_set_event_handler(source, block);
    dispatch_resume(source);
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSString *diskPath = nil;
        NSString *seedDiskPath = nil;
        NSString *nvramPath = nil;
        NSString *machineIDPath = nil;
        NSString *serialPath = nil;
        NSString *unixSocketPath = nil;
        NSString *natMAC = nil;
        BOOL useStdioConsole = NO;
        NSUInteger memoryMiB = 4096;
        NSUInteger cpuCount = 4;
        uint32_t vsockPort = 5000;
        BOOL enableNAT = NO;

        for (int i = 1; i < argc; i++) {
            NSString *arg = [NSString stringWithUTF8String:argv[i]];
            if ([arg isEqualToString:@"--help"] || [arg isEqualToString:@"-h"]) {
                print_usage(stdout);
                return 0;
            }
            if ([arg isEqualToString:@"--disk"] && i + 1 < argc) {
                diskPath = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--seed-disk"] && i + 1 < argc) {
                seedDiskPath = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--nvram"] && i + 1 < argc) {
                nvramPath = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--machine-id"] && i + 1 < argc) {
                machineIDPath = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--serial-log"] && i + 1 < argc) {
                serialPath = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--stdio-console"]) {
                useStdioConsole = YES;
            } else if ([arg isEqualToString:@"--unix-socket"] && i + 1 < argc) {
                unixSocketPath = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--nat-mac"] && i + 1 < argc) {
                natMAC = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--vsock-port"] && i + 1 < argc) {
                vsockPort = (uint32_t)strtoul(argv[++i], NULL, 10);
            } else if ([arg isEqualToString:@"--memory-mib"] && i + 1 < argc) {
                memoryMiB = (NSUInteger)strtoul(argv[++i], NULL, 10);
            } else if ([arg isEqualToString:@"--cpu-count"] && i + 1 < argc) {
                cpuCount = (NSUInteger)strtoul(argv[++i], NULL, 10);
            } else if ([arg isEqualToString:@"--enable-nat"]) {
                enableNAT = YES;
            } else {
                fprintf(stderr, "unknown argument: %s\n", argv[i]);
                return 2;
            }
        }

        if (diskPath.length == 0 || unixSocketPath.length == 0) {
            print_usage(stderr);
            return 2;
        }
        if (serialPath.length > 0 && useStdioConsole) {
            fprintf(stderr, "--serial-log and --stdio-console are mutually exclusive\n");
            return 2;
        }
        if (!enableNAT && natMAC.length > 0) {
            fprintf(stderr, "--nat-mac requires --enable-nat\n");
            return 2;
        }
        if (vsockPort == 0) {
            fprintf(stderr, "--vsock-port must be greater than zero\n");
            return 2;
        }
        if (memoryMiB == 0 || cpuCount == 0) {
            fprintf(stderr, "--memory-mib and --cpu-count must be greater than zero\n");
            return 2;
        }

        VzVsockRunner *runner = [[VzVsockRunner alloc] initWithUnixSocketPath:unixSocketPath vsockPort:vsockPort];
        NSError *error = nil;
        if (![runner startWithDiskPath:diskPath
                          seedDiskPath:seedDiskPath
                             nvramPath:nvramPath
                         machineIDPath:machineIDPath
                            serialPath:serialPath
                      useStdioConsole:useStdioConsole
                            memoryMiB:memoryMiB
                             cpuCount:cpuCount
                               natMAC:natMAC
                            enableNAT:enableNAT
                                error:&error]) {
            fprintf(stderr, "failed to start Vz VM: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        install_signal_handler(SIGINT, ^{
            exit(0);
        });
        install_signal_handler(SIGTERM, ^{
            exit(0);
        });

        dispatch_main();
    }
}
