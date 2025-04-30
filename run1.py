import socket

DEVICE_IP = "192.168.0.102"  # apna device IP daalo
DEVICE_PORT = 5005  # apna device Port daalo


def listen_for_logins():
    try:
        print(f"🔄 Connecting to {DEVICE_IP}:{DEVICE_PORT}...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((DEVICE_IP, DEVICE_PORT))
        print("✅ Connected to device.")
        print(s)

        print("👂 Listening for login events... (Press Ctrl+C to stop)")
        while True:
            print(s)
            try:
                # print(s.port)
                # print(s.getsockname())
                # print(s.getpeername())
                # print(s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR))
                # print(f"🌐 Current Socket URL: {s.getsockname()[0]}:{s.getsockname()[1]}")
                data = s.recv(4096)  # bada buffer
                if data:
                    print("\n📥 New Data Received!")
                    print("Hex Format:", data.hex())
                    try:
                        print("ASCII Format:", data.decode("utf-8", errors="ignore"))
                    except Exception as e:
                        print("🚫 Error decoding to ASCII:", e)
                else:
                    print("😶 No data, device might be idle.")
            except socket.timeout:
                print("⏳ Waiting for data...")
                print(s)
            except KeyboardInterrupt:
                print("\n🛑 Stopping listening...")
                break

        s.close()
        print("🔌 Disconnected from device.")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    listen_for_logins()
