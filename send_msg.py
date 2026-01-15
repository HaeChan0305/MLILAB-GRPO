import requests

def send_hit_msg():
  SLACK_TOKEN = None
  USER_ID = "U079DQF6JCW"

  response = requests.post(
      'https://slack.com/api/chat.postMessage',
      headers={'Authorization': f'Bearer {SLACK_TOKEN}'},
      json={
          'channel': USER_ID,
          'text': "Process Killed"
      }
  )
  
send_hit_msg()