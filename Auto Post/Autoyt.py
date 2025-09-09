#Import Required Libraries.
import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
from googleapiclient.http import MediaFileUpload

#Change Your Specific Client Deatils In 'client_secrets.json' File.
CLIENT_SECRET_FILE = 'client_secrets.json'
API_NAME = 'YouTube'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

def authenticate_and_upload_video(video_path, title, description, privacy_status):
    # Set up the OAuth 2.0 flow for YouTube API.
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    credentials = flow.run_console

    # Create a YouTube service instance.
    youtube = googleapiclient.discovery.build(API_NAME, API_VERSION, credentials=credentials)

    # Upload the video.
    request_body = {
        'snippet': {
            'title': title,
            'description': description
        },
        'status': {
            'privacyStatus': privacy_status
        }
    }

    media = MediaFileUpload(video_path)
    response = youtube.videos().insert(
        part='snippet,status',
        body=request_body,
        media_body=media
    ).execute()

    print(f'Video uploaded! Video ID: {response["id"]}')a

if __name__ == '__main__':
    video_path = 'example.mov'
    video_title = 'Your Video Title'
    video_description = 'Your Video Description'
    video_privacy_status = 'public'  # Change to 'private' or 'unlisted' as needed

    authenticate_and_upload_video(video_path, video_title, video_description, video_privacy_status)
 