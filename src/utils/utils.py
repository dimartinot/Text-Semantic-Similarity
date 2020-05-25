from datetime import datetime

def _debug(debug, msg, msg_type="INFO"):
    """
    Method printing debug messages
    """
    if (debug):
        if (isinstance(msg, list)):
            for m in msg:
                date = datetime.now().strftime("%m-%d %H:%M:%S")
                print(f"[{msg_type}] - {date} - {m}")
        else:
            date = datetime.now().strftime("%m-%d %H:%M:%S")
            print(f"[{msg_type}] - {date} - {msg}")