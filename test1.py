import asyncio
import socket
import websockets

DEVICE_IP = "192.168.0.100"
DEVICE_PORT = 5005

WS_SERVER_HOST = "0.0.0.0"
WS_SERVER_PORT = 8765


# Connect to fingerprint device over TCP
def connect_to_device():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((DEVICE_IP, DEVICE_PORT))
    return s


# Coroutine to handle WebSocket connections
async def websocket_handler(websocket):
    print(f"Client connected: {websocket.remote_address}")

    device_socket = connect_to_device()

    async def device_to_websocket():
        while True:
            data = device_socket.recv(1024)
            if not data:
                break
            await websocket.send(data.hex())  # send data to client

    async def websocket_to_device():
        async for message in websocket:
            device_socket.sendall(bytes.fromhex(message))  # send data to device

    await asyncio.gather(device_to_websocket(), websocket_to_device())

    device_socket.close()


# Main: Start WebSocket Server
async def main():
    async with websockets.serve(websocket_handler, WS_SERVER_HOST, WS_SERVER_PORT):
        print(f"WebSocket Server running at ws://{WS_SERVER_HOST}:{WS_SERVER_PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
