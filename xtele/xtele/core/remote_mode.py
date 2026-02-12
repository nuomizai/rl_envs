import time
import zmq
import threading
import pickle
import numpy as np

from xtele.core.integrate_module import TeleCore


np.set_printoptions(6)
np.set_printoptions(suppress=True)

TELE_COMM_PORT = 4399


class TeleServer:
    def __init__(self):
        self.tele_agent = TeleCore()
        tmp = self.tele_agent.act_dict()
        print("First try: ", tmp)
        host = "127.0.0.1"
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self.addr = f"tcp://{host}:{TELE_COMM_PORT}"
        self._socket.bind(self.addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        while not self._stop_event.is_set():
            start_time = time.time()
            try:
                _ = self._socket.recv()
                self._socket.send(pickle.dumps(self.tele_agent.act()))
                print(
                    f"Freq on {TELE_COMM_PORT} is {round(100 / (time.time() - start_time)) / 100} "
                )
            except zmq.Again:
                print("Tele Act Dict Timeout")

    def __del__(self) -> None:
        self._stop_event.set()


class TeleClient:
    def __init__(self):
        port = 4399
        host = "127.0.0.1"
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def act_dict(self):
        send_message = pickle.dumps([])
        self._socket.send(send_message)
        return pickle.loads(self._socket.recv())

    def act(self):
        return self.act_dict()
