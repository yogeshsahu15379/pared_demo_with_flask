from zk import ZK, const
import time

DEVICE_IP = "192.168.0.102"
DEVICE_PORT = 5005  # or 4370, depending on your model


def poll_device():
    zk = ZK(
        DEVICE_IP,
        port=DEVICE_PORT,
        timeout=5,
        password=0,
        force_udp=False,
        ommit_ping=False,
    )
    conn = zk.connect()
    try:
        # temporarily disable device so we can safely read its logs
        conn.disable_device()
        logs = conn.get_attendance()  # fetch all attendance records
        for log in logs:
            print(
                f"👤 User {log.user_id} scanned at {log.timestamp} (status: {log.status})"
            )
        # (you could also clear logs if desired: conn.clear_attendance())
    finally:
        conn.enable_device()  # re-enable normal operations
        conn.disconnect()


if __name__ == "__main__":
    print("🔄 Starting polling loop (every 10s)...")
    while True:
        poll_device()
        time.sleep(10)
