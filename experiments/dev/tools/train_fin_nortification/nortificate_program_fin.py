from slack_post import send_message_to_slack
import socket
import argparse
import datetime


def main(args):
    hostname = socket.gethostname()
    dt_now = datetime.datetime.now ( )
    time_str = f"{dt_now.year}年{dt_now.month}月{dt_now.day}日, {dt_now.hour}:{dt_now.minute}:{dt_now.second}"
    message= f"プログラムが終了しました!\n============================\nサーバー名 : {hostname}\n終了時刻   : {time_str}\n============================\n"

    message += args.message

    
    send_message_to_slack(message)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--message", help="input additional message", type = str, default="")
    
    args = parser.parse_args()
    main(args)
    
    
