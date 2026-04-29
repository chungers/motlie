#import <Foundation/Foundation.h>
#import <Virtualization/Virtualization.h>

#include <dispatch/dispatch.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static char g_net_socket_local_path[sizeof(((struct sockaddr_un *)0)->sun_path)] = {0};
static const NSUInteger kMotlieVzNetworkMTU = 1500;

static void cleanup_net_socket_local_path(void) {
    if (g_net_socket_local_path[0] != '\0') {
        unlink(g_net_socket_local_path);
        g_net_socket_local_path[0] = '\0';
    }
}

static void print_usage(FILE *stream) {
    fprintf(stream,
            "usage: vz-vsock-runner --disk <path> --unix-socket <path> "
            "[--seed-disk <path>] [--nvram <path>] [--machine-id <path>] "
            "[--serial-log <path> | --stdio-console] [--nat-mac <mac>] [--net-mac <mac>] "
            "[--net-backend-socket <path>] "
            "[--vsock-forward <port:path>] "
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

static int bind_unix_datagram_socket(void) {
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("socket(AF_UNIX, SOCK_DGRAM)");
        return -1;
    }

    int sndbuf = 256 * 1024;
    int rcvbuf = 1024 * 1024;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) != 0) {
        perror("setsockopt(SO_SNDBUF)");
        close(fd);
        return -1;
    }
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) != 0) {
        perror("setsockopt(SO_RCVBUF)");
        close(fd);
        return -1;
    }

    NSString *localPath = [NSString stringWithFormat:@"/tmp/motlie-vz-net-%d.sock", getpid()];
    unlink(localPath.fileSystemRepresentation);
    struct sockaddr_un localAddr = {0};
    localAddr.sun_family = AF_UNIX;
    const char *clocal = [localPath fileSystemRepresentation];
    if (strlen(clocal) >= sizeof(localAddr.sun_path)) {
        fprintf(stderr, "local unix datagram path too long: %s\n", clocal);
        close(fd);
        return -1;
    }
    strncpy(localAddr.sun_path, clocal, sizeof(localAddr.sun_path) - 1);
    if (bind(fd, (struct sockaddr *)&localAddr, sizeof(localAddr)) != 0) {
        perror("bind(AF_UNIX, SOCK_DGRAM)");
        close(fd);
        return -1;
    }
    strlcpy(g_net_socket_local_path, clocal, sizeof(g_net_socket_local_path));
    atexit(cleanup_net_socket_local_path);
    return fd;
}

static int tune_datagram_fd(int fd) {
    int sndbuf = 256 * 1024;
    int rcvbuf = 1024 * 1024;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf)) != 0) {
        perror("setsockopt(SO_SNDBUF)");
        return -1;
    }
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) != 0) {
        perror("setsockopt(SO_RCVBUF)");
        return -1;
    }
    return 0;
}

static void start_datagram_bridge(const char *label, int srcFD, int dstFD) {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        uint8_t buffer[65536];
        uint64_t count = 0;
        while (1) {
            ssize_t n = recv(srcFD, buffer, sizeof(buffer), 0);
            if (n == 0) {
                break;
            }
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            count += 1;
            if (count <= 10 || count % 100 == 0) {
                fprintf(stderr, "net bridge %s frame %llu (%zd bytes)\n", label, count, n);
            }
            while (1) {
                ssize_t sent = send(dstFD, buffer, (size_t)n, 0);
                if (sent >= 0) {
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                if (errno == ENOBUFS || errno == EAGAIN) {
                    usleep(1000);
                    continue;
                }
                perror("send(AF_UNIX, SOCK_DGRAM)");
                goto bridge_done;
            }
        }
bridge_done:
        shutdown(srcFD, SHUT_RDWR);
        shutdown(dstFD, SHUT_RDWR);
    });
}

static void start_datagram_sendto_bridge(const char *label, int srcFD, int dstFD, NSString *dstPath) {
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        uint8_t buffer[65536];
        uint64_t count = 0;
        struct sockaddr_un dstAddr = {0};
        dstAddr.sun_family = AF_UNIX;
        const char *cdst = [dstPath fileSystemRepresentation];
        if (strlen(cdst) >= sizeof(dstAddr.sun_path)) {
            fprintf(stderr, "unix datagram backend path too long: %s\n", cdst);
            return;
        }
        strncpy(dstAddr.sun_path, cdst, sizeof(dstAddr.sun_path) - 1);

        while (1) {
            ssize_t n = recv(srcFD, buffer, sizeof(buffer), 0);
            if (n == 0) {
                break;
            }
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            count += 1;
            if (count <= 10 || count % 100 == 0) {
                fprintf(stderr, "net bridge %s frame %llu (%zd bytes)\n", label, count, n);
            }
            while (1) {
                ssize_t sent = sendto(dstFD, buffer, (size_t)n, 0, (struct sockaddr *)&dstAddr, sizeof(dstAddr));
                if (sent >= 0) {
                    break;
                }
                if (errno == EINTR) {
                    continue;
                }
                if (errno == ENOBUFS || errno == EAGAIN) {
                    usleep(1000);
                    continue;
                }
                perror("sendto(AF_UNIX, SOCK_DGRAM)");
                goto bridge_done;
            }
        }
bridge_done:
        shutdown(srcFD, SHUT_RDWR);
        shutdown(dstFD, SHUT_RDWR);
    });
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
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSString *> *socketForwardPaths;
@property(nonatomic, strong) dispatch_queue_t vmQueue;
@property(nonatomic, strong) NSMutableArray<NSNumber *> *networkBridgeFDs;
- (instancetype)initWithSocketForwardPaths:(NSDictionary<NSNumber *, NSString *> *)socketForwardPaths;
- (nullable NSString *)unixSocketPathForPort:(uint32_t)port;
- (BOOL)startWithDiskPath:(NSString *)diskPath
               seedDiskPath:(nullable NSString *)seedDiskPath
                nvramPath:(nullable NSString *)nvramPath
            machineIDPath:(nullable NSString *)machineIDPath
               serialPath:(nullable NSString *)serialPath
         useStdioConsole:(BOOL)useStdioConsole
               memoryMiB:(NSUInteger)memoryMiB
                cpuCount:(NSUInteger)cpuCount
                  natMAC:(nullable NSString *)natMAC
                  netMAC:(nullable NSString *)netMAC
        netBackendSocketPath:(nullable NSString *)netBackendSocketPath
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

    NSString *unixSocketPath = [self.runner unixSocketPathForPort:connection.destinationPort];
    if (unixSocketPath.length == 0) {
        fprintf(stderr, "no unix socket configured for guest virtio-socket port %u\n",
                connection.destinationPort);
        return NO;
    }

    int unixFD = connect_unix_socket(unixSocketPath);
    if (unixFD < 0) {
        fprintf(stderr, "failed to connect host unix socket for port %u: %s\n",
                connection.destinationPort,
                unixSocketPath.UTF8String);
        return NO;
    }

    BridgeSession *session = [[BridgeSession alloc] initWithConnection:connection unixFD:unixFD runner:self.runner];
    [self.runner addSession:session];
    [session start];
    fprintf(stderr, "accepted guest virtio-socket connection on port %u -> %s\n",
            connection.destinationPort, unixSocketPath.UTF8String);
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

- (instancetype)initWithSocketForwardPaths:(NSDictionary<NSNumber *, NSString *> *)socketForwardPaths {
    self = [super init];
    if (self) {
        _socketForwardPaths = [socketForwardPaths mutableCopy];
        _sessions = [NSMutableSet set];
        _vmQueue = dispatch_queue_create("motlie.vmm.v1_45.vz_runner", DISPATCH_QUEUE_SERIAL);
        _networkBridgeFDs = [NSMutableArray array];
    }
    return self;
}

- (nullable NSString *)unixSocketPathForPort:(uint32_t)port {
    return self.socketForwardPaths[@(port)];
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
                  netMAC:(nullable NSString *)netMAC
        netBackendSocketPath:(nullable NSString *)netBackendSocketPath
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
    NSString *loggedNetMAC = @"(disabled)";

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

    NSMutableArray<VZNetworkDeviceConfiguration *> *networkDevices = [NSMutableArray array];

    if (enableNAT) {
        VZVirtioNetworkDeviceConfiguration *net = [[VZVirtioNetworkDeviceConfiguration alloc] init];
        net.attachment = [[VZNATNetworkDeviceAttachment alloc] init];
        VZMACAddress *macAddress = nil;
        if (natMAC.length > 0) {
            macAddress = [[VZMACAddress alloc] initWithString:natMAC];
            if (!macAddress) {
                if (error) {
                    NSString *message = [NSString stringWithFormat:@"invalid NAT MAC address: %@", natMAC];
                    *error = [NSError errorWithDomain:@"motlie.vnet.v1_25.vz"
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
        [networkDevices addObject:net];
    }

    if (netBackendSocketPath.length > 0) {
        int netBackendFD = bind_unix_datagram_socket();
        if (netBackendFD < 0) {
            if (error) {
                NSString *message = [NSString stringWithFormat:@"failed to bind unix datagram network backend socket for %@", netBackendSocketPath];
                *error = [NSError errorWithDomain:@"motlie.vnet.v1_25.vz"
                                             code:4
                                         userInfo:@{NSLocalizedDescriptionKey: message}];
            }
            return NO;
        }

        int netPair[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_DGRAM, 0, netPair) != 0) {
            perror("socketpair(AF_UNIX, SOCK_DGRAM)");
            close(netBackendFD);
            if (error) {
                *error = [NSError errorWithDomain:@"motlie.vnet.v1_25.vz"
                                             code:4
                                         userInfo:@{NSLocalizedDescriptionKey: @"failed to create guest network socketpair"}];
            }
            return NO;
        }
        if (tune_datagram_fd(netPair[0]) != 0 || tune_datagram_fd(netPair[1]) != 0) {
            close(netPair[0]);
            close(netPair[1]);
            close(netBackendFD);
            if (error) {
                *error = [NSError errorWithDomain:@"motlie.vnet.v1_25.vz"
                                             code:4
                                         userInfo:@{NSLocalizedDescriptionKey: @"failed to tune guest network socketpair"}];
            }
            return NO;
        }

        start_datagram_sendto_bridge("guest->backend", netPair[1], netBackendFD, netBackendSocketPath);
        start_datagram_bridge("backend->guest", netBackendFD, netPair[1]);
        [self.networkBridgeFDs addObject:@(netPair[1])];
        [self.networkBridgeFDs addObject:@(netBackendFD)];

        NSFileHandle *netFile = [[NSFileHandle alloc] initWithFileDescriptor:netPair[0] closeOnDealloc:YES];
        VZFileHandleNetworkDeviceAttachment *attachment = [[VZFileHandleNetworkDeviceAttachment alloc] initWithFileHandle:netFile];
        attachment.maximumTransmissionUnit = kMotlieVzNetworkMTU;
        VZVirtioNetworkDeviceConfiguration *net = [[VZVirtioNetworkDeviceConfiguration alloc] init];
        net.attachment = attachment;
        VZMACAddress *macAddress = nil;
        if (netMAC.length > 0) {
            macAddress = [[VZMACAddress alloc] initWithString:netMAC];
            if (!macAddress) {
                if (error) {
                    NSString *message = [NSString stringWithFormat:@"invalid network MAC address: %@", netMAC];
                    *error = [NSError errorWithDomain:@"motlie.vnet.v1_25.vz"
                                                 code:5
                                             userInfo:@{NSLocalizedDescriptionKey: message}];
                }
                return NO;
            }
        } else {
            macAddress = [VZMACAddress randomLocallyAdministeredAddress];
        }
        net.MACAddress = macAddress;
        loggedNetMAC = macAddress.string;
        [networkDevices addObject:net];
    }

    if (networkDevices.count > 0) {
        config.networkDevices = networkDevices;
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
        for (NSNumber *portNumber in self.socketForwardPaths) {
            VZVirtioSocketListener *listener = [[VZVirtioSocketListener alloc] init];
            listener.delegate = self.socketDelegate;
            [socketDevice setSocketListener:listener forPort:portNumber.unsignedIntValue];
        }

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

    NSMutableArray<NSString *> *forwardings = [NSMutableArray array];
    NSArray<NSNumber *> *sortedPorts = [[self.socketForwardPaths allKeys] sortedArrayUsingSelector:@selector(compare:)];
    for (NSNumber *portNumber in sortedPorts) {
        NSString *path = self.socketForwardPaths[portNumber];
        [forwardings addObject:[NSString stringWithFormat:@"%@->%@", portNumber, path]];
    }

    fprintf(stderr, "vz-vsock-runner started: disk=%s seed=%s forwards=%s nat-mac=%s net-mac=%s\n",
            diskPath.UTF8String,
            seedDiskPath.length > 0 ? seedDiskPath.UTF8String : "(none)",
            [[forwardings componentsJoinedByString:@","] UTF8String],
            loggedNatMAC.UTF8String,
            loggedNetMAC.UTF8String);
    return YES;
}

@end

static void install_signal_handler(int signum, dispatch_block_t block) {
    signal(signum, SIG_IGN);
    dispatch_source_t source = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, signum, 0, dispatch_get_main_queue());
    dispatch_source_set_event_handler(source, block);
    dispatch_resume(source);
}

// @opus47-mac 2026-04-26 -- request_clean_stop drives VZ shutdown via the
// framework's stopWithCompletionHandler so destructors run, the disk
// synchronization mode (VZDiskImageSynchronizationModeFsync) gets to flush,
// and the kernel fd flush completes before the process exits. The previous
// SIGINT/SIGTERM handlers called exit(0) directly, which under concurrent
// multi-guest load could interrupt writes mid-flight (issue #215 / PR #212
// finding #6).
//
// dispatch_once guards against double-stop on rapid Ctrl-C; the deadline
// timer is a fallback for the rare case where stopWithCompletionHandler
// hangs (e.g. framework deadlock). _exit avoids running atexit handlers
// in that already-broken state.
static dispatch_once_t s_stopOnce;
static const int64_t STOP_DEADLINE_NS = 5LL * NSEC_PER_SEC;

static void request_clean_stop(VzVsockRunner *runner, int signum) {
    dispatch_once(&s_stopOnce, ^{
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, STOP_DEADLINE_NS),
                       dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
            fprintf(stderr,
                    "vz-vsock-runner: stop deadline elapsed (sig=%d); forcing exit\n",
                    signum);
            _exit(1);
        });
        dispatch_async(runner.vmQueue, ^{
            [runner.vm stopWithCompletionHandler:^(NSError * _Nullable error) {
                if (error) {
                    fprintf(stderr,
                            "vz-vsock-runner: stop failed (sig=%d): %s\n",
                            signum,
                            error.localizedDescription.UTF8String);
                    exit(1);
                }
                exit(0);
            }];
        });
    });
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
        NSString *netMAC = nil;
        NSString *netBackendSocketPath = nil;
        NSMutableDictionary<NSNumber *, NSString *> *socketForwardPaths = [NSMutableDictionary dictionary];
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
            } else if ([arg isEqualToString:@"--vsock-forward"] && i + 1 < argc) {
                NSString *spec = [NSString stringWithUTF8String:argv[++i]];
                NSRange sep = [spec rangeOfString:@":"];
                if (sep.location == NSNotFound || sep.location == 0 || sep.location == spec.length - 1) {
                    fprintf(stderr, "invalid --vsock-forward '%s'; expected <port>:<unix-socket>\n", spec.UTF8String);
                    return 2;
                }
                NSString *portText = [spec substringToIndex:sep.location];
                NSString *pathText = [spec substringFromIndex:sep.location + 1];
                uint32_t port = (uint32_t)strtoul(portText.UTF8String, NULL, 10);
                if (port == 0 || pathText.length == 0) {
                    fprintf(stderr, "invalid --vsock-forward '%s'; expected <port>:<unix-socket>\n", spec.UTF8String);
                    return 2;
                }
                socketForwardPaths[@(port)] = pathText;
            } else if ([arg isEqualToString:@"--nat-mac"] && i + 1 < argc) {
                natMAC = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--net-mac"] && i + 1 < argc) {
                netMAC = [NSString stringWithUTF8String:argv[++i]];
            } else if ([arg isEqualToString:@"--net-backend-socket"] && i + 1 < argc) {
                netBackendSocketPath = [NSString stringWithUTF8String:argv[++i]];
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

        if (diskPath.length == 0) {
            print_usage(stderr);
            return 2;
        }
        if (socketForwardPaths.count == 0) {
            if (unixSocketPath.length == 0) {
                print_usage(stderr);
                return 2;
            }
            socketForwardPaths[@(vsockPort)] = unixSocketPath;
        }
        if (serialPath.length > 0 && useStdioConsole) {
            fprintf(stderr, "--serial-log and --stdio-console are mutually exclusive\n");
            return 2;
        }
        if (!enableNAT && natMAC.length > 0) {
            fprintf(stderr, "--nat-mac requires --enable-nat\n");
            return 2;
        }
        if (enableNAT && netBackendSocketPath.length > 0) {
            fprintf(stderr, "--enable-nat and --net-backend-socket are mutually exclusive\n");
            return 2;
        }
        if (netMAC.length > 0 && netBackendSocketPath.length == 0) {
            fprintf(stderr, "--net-mac requires --net-backend-socket\n");
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

        VzVsockRunner *runner = [[VzVsockRunner alloc] initWithSocketForwardPaths:socketForwardPaths];
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
                                  netMAC:netMAC
                        netBackendSocketPath:netBackendSocketPath
                               enableNAT:enableNAT
                                   error:&error]) {
            fprintf(stderr, "failed to start Vz VM: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        install_signal_handler(SIGINT, ^{
            request_clean_stop(runner, SIGINT);
        });
        install_signal_handler(SIGTERM, ^{
            request_clean_stop(runner, SIGTERM);
        });

        dispatch_main();
    }
}
