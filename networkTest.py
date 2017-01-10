import time
import websocket


def main():
    address = get_address_from_config('../config.txt')
    ws = websocket.WebSocketApp(address, on_close = on_close, on_error = on_error, on_message = on_message)
    ws.on_open = on_open
    ws.run_forever()
    time.sleep(30)
    for i in range(10):
        ws.send('REQUEST')
        time.sleep(10)
    ws.send('CLOSE')
    ws.close()


def get_address_from_config(config_file):
    try:
        with open(config_file, 'r') as file:
            for line in file.readlines():
                line.strip()
                if line and len(line) > 0:
                    pair = line.split("=")
                    if pair[0] == 'EmotionAddress':
                        return pair[1]
        return 0
    except IOError as err:
        print err.message


def on_open(ws):
    print 'Connection Opened'


def on_close(ws):
    print 'Connection Closed'


def on_error(ws, err):
    print err.message


def on_message(ws, data):
    print data

if __name__ == '__main__':
    main()