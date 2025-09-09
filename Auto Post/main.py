import tweepy

# Authenticate to Twitter
def authenticate_twitter(api_key, api_key_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api

# Function to tweet
def tweet(api, message):
    try:
        api.update_status(message)
        print(f"Successfully tweeted: {message}")
    except Exception as e:
        print(f"Error occurred: {e}")

# Twitter API credentials
api_key = 'your_api_key'
api_key_secret = 'your_api_secret_key'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

# Authenticate and tweet
api = authenticate_twitter(api_key, api_key_secret, access_token, access_token_secret)

# Post a tweet
tweet(api, "Hello, this is an automated tweet from my Python script!")
