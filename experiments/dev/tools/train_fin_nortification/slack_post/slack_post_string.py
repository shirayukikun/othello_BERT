import requests
import json
import os
from pathlib import Path

def send_message_to_slack(message):

    with open(os.path.join(os.path.dirname(__file__), "config.json"), mode="r") as f:
        config_dict = json.load(f)

    try:
        token = config_dict["token"]
        channel = config_dict["channel"]
        username = config_dict["username"]
    except KeyError:
        path = Path(__file__).parent / "config.json"
        print(f"Specify slack bot token, channel name and user name to {path}")
        return

    url = f"https://slack.com/api/chat.postMessage"

    payload = {
        "token" : token,
        "channel" : channel,
        "text" : message,
        "username" : username,
        "attachments": ""
        }

    response = requests.post(url, params=payload)
    if response.status_code == 200:
        print("succeed to send message")
    else:
        print("failed to send message")


if __name__ == "__main__":
    send_message_to_slack("テストメッセージ")
    
