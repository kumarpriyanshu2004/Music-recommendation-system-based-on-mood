from tkinter import ttk

from PIL import Image, ImageTk
from PIL.ImageTk import PhotoImage
from sklearn.metrics import silhouette_score

import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials

from emotion_video_classifier import emotion_testing
import tkinter as tk
from tkinter import messagebox

cid ="e45c35b36db446a0a4baffdc99b1272e"
secret = "9373a03a8e2c4ecf9bea31f8abfb8307"
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

root = tk.Tk()
root.title('CREDENTIALS')
root.geometry("600x400")
root.configure(bg='black')
name1 = tk.StringVar()
image_path = r"C:\Users\ASUS\Desktop\Music-recommendation-system\images\musicback.jpg"
file = Image.open(image_path)
photo = PhotoImage(file)
l = tk.Label(root, image=photo)
l.image = photo  # just keeping a reference
l.grid()


def submit():
    global name
    name = name_entry.get()
    messagebox.showinfo("Information", "Wait for sometime for us to create Playlists")
    root.destroy()


name_label = tk.Label(root, text='Enter Name of Artist',
                      font=('calibre',
                            10, 'bold'))

name_entry = tk.Entry(root,
                      textvariable=name1, font=('calibre', 10, 'normal'))

sub_btn = tk.Button(root, text='Submit',
                    command=submit)
print("roger")
name_label.grid(row=0, column=0)
name_entry.grid(row=3, column=0)
sub_btn.grid(row=5, column=0)
root.mainloop()

result = sp.search(name)  # search query

artist_uri = result['tracks']['items'][0]['artists'][0]['uri']
# Pull all of the artist's albums
sp_albums = sp.artist_albums(artist_uri, album_type='album')
# Store artist's albums' names' and uris in separate lists
album_names = []
album_uris = []
for i in range(len(sp_albums['items'])):
    album_names.append(sp_albums['items'][i]['name'])
    album_uris.append(sp_albums['items'][i]['uri'])


def albumSongs(uri):
    album = uri  # assign album uri to a_name
    spotify_albums[album] = {}  # Creates dictionary for that specific album
    # Create keys-values of empty lists inside nested dictionary for album
    spotify_albums[album]['album'] = []  # create empty list
    spotify_albums[album]['track_number'] = []
    spotify_albums[album]['id'] = []
    spotify_albums[album]['name'] = []
    spotify_albums[album]['uri'] = []
    tracks = sp.album_tracks(album)  # pull data on album tracks
    for n in range(len(tracks['items'])):  # for each song track
        spotify_albums[album]['album'].append(album_names[album_count])  # append album name tracked via album_count
        spotify_albums[album]['track_number'].append(tracks['items'][n]['track_number'])
        spotify_albums[album]['id'].append(tracks['items'][n]['id'])
        spotify_albums[album]['name'].append(tracks['items'][n]['name'])
        spotify_albums[album]['uri'].append(tracks['items'][n]['uri'])


spotify_albums = {}
album_count = 0
for i in album_uris:  # each album
    albumSongs(i)
    print("Album " + str(album_names[album_count]) + " songs has been added to spotify_albums dictionary")
    album_count += 1  # Updates album count once all tracks have been added

print("roger1")
# def audio_features(album):
#     # Add new key-values to store audio features
#     spotify_albums[album]['acousticness'] = []
#     spotify_albums[album]['danceability'] = []
#     spotify_albums[album]['energy'] = []
#     spotify_albums[album]['instrumentalness'] = []
#     spotify_albums[album]['liveness'] = []
#     spotify_albums[album]['loudness'] = []
#     spotify_albums[album]['speechiness'] = []
#     spotify_albums[album]['tempo'] = []
#     spotify_albums[album]['valence'] = []
#     spotify_albums[album]['popularity'] = []
#     # create a track counter
#     track_count = 0
#     for track in spotify_albums[album]['uri']:
#         # pull audio features per track
#         features = sp.audio_features(track)

#         # Append to relevant key-value
#         spotify_albums[album]['acousticness'].append(features[0]['acousticness'])
#         spotify_albums[album]['danceability'].append(features[0]['danceability'])
#         spotify_albums[album]['energy'].append(features[0]['energy'])
#         spotify_albums[album]['instrumentalness'].append(features[0]['instrumentalness'])
#         spotify_albums[album]['liveness'].append(features[0]['liveness'])
#         spotify_albums[album]['loudness'].append(features[0]['loudness'])
#         spotify_albums[album]['speechiness'].append(features[0]['speechiness'])
#         spotify_albums[album]['tempo'].append(features[0]['tempo'])
#         spotify_albums[album]['valence'].append(features[0]['valence'])
#         # popularity is stored elsewhere
#         pop = sp.track(track)
#         spotify_albums[album]['popularity'].append(pop['popularity'])

#         track_count += 1
def audio_features(album):
    # Add new key-values to store audio features
    feature_keys = [
        'acousticness', 'danceability', 'energy', 'instrumentalness',
        'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity'
    ]
    for key in feature_keys:
        spotify_albums[album][key] = []

    # Create a track counter
    track_count = 0

    for track in spotify_albums[album]['uri']:
        try:
            # Pull audio features per track
            features = sp.audio_features(track)
            if features and features[0]:  # Ensure valid response
                spotify_albums[album]['acousticness'].append(features[0].get('acousticness', None))
                spotify_albums[album]['danceability'].append(features[0].get('danceability', None))
                spotify_albums[album]['energy'].append(features[0].get('energy', None))
                spotify_albums[album]['instrumentalness'].append(features[0].get('instrumentalness', None))
                spotify_albums[album]['liveness'].append(features[0].get('liveness', None))
                spotify_albums[album]['loudness'].append(features[0].get('loudness', None))
                spotify_albums[album]['speechiness'].append(features[0].get('speechiness', None))
                spotify_albums[album]['tempo'].append(features[0].get('tempo', None))
                spotify_albums[album]['valence'].append(features[0].get('valence', None))
            else:
                # Append `None` if features are not available
                for key in feature_keys[:-1]:  # Exclude 'popularity'
                    spotify_albums[album][key].append(None)

            # Popularity is stored in the track data
            pop = sp.track(track)
            spotify_albums[album]['popularity'].append(pop.get('popularity', None))

            track_count += 1
        except Exception as e:
            # Handle exceptions (e.g., API rate limits, invalid responses)
            print(f"Error fetching features for track {track}: {e}")
            # Append `None` for all keys to maintain data integrity
            for key in feature_keys:
                spotify_albums[album][key].append(None)
print("roger2")

# import time
# import numpy as np

# sleep_min = 2
# sleep_max = 5
# start_time = time.time()
# request_count = 0
# for i in spotify_albums:
#     audio_features(i)
#     request_count += 1
#     if request_count % 5 == 0:
#         print(str(request_count) + " playlists completed")
#         time.sleep(np.random.uniform(sleep_min, sleep_max))
#         print('Loop : {}'.format(request_count))
#         print('Elapsed Time: {} seconds'.format(time.time() - start_time))
import time
import numpy as np

# Minimum and maximum sleep times between API requests to avoid rate-limiting
sleep_min = 2
sleep_max = 5

# Track the start time and the number of requests made
start_time = time.time()
request_count = 0

# Iterate over the albums in spotify_albums
for i, album in enumerate(spotify_albums.keys()):
    try:
        # Call the function to fetch audio features
        audio_features(album)
        request_count += 1

        # Print progress every 5 requests
        if request_count % 5 == 0:
            print(f"{request_count} playlists completed")
            elapsed_time = time.time() - start_time
            print(f"Loop: {request_count}")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
            
            # Sleep for a random duration to avoid rate-limiting
            time.sleep(np.random.uniform(sleep_min, sleep_max))
    except Exception as e:
        print(f"Error processing album {album} at index {i}: {e}")
        continue  # Skip to the next album if an error occurs


dic_df = {}
dic_df['album'] = []
dic_df['track_number'] = []
dic_df['id'] = []
dic_df['name'] = []
dic_df['uri'] = []
dic_df['acousticness'] = []
dic_df['danceability'] = []
dic_df['energy'] = []
dic_df['instrumentalness'] = []
dic_df['liveness'] = []
dic_df['loudness'] = []
dic_df['speechiness'] = []
dic_df['tempo'] = []
dic_df['valence'] = []
dic_df['popularity'] = []
for album in spotify_albums:
    for feature in spotify_albums[album]:
        dic_df[feature].extend(spotify_albums[album][feature])

length = len(dic_df['album'])

# data = pd.DataFrame.from_dict(dic_df)
# data.drop_duplicates(inplace=True, subset=['name'])
# name = data['name']
# df = pd.read_csv(r"C:\Users\ASUS\Desktop\Music-recommendation-system\Spotify Dataset Analysis\fer2013.csv")
# df.drop_duplicates(inplace=True, subset=['name'])
# name = df['name']
# data1 = pd.concat([data, df], ignore_index=True)

# name = data1['name']
# Convert dictionary to DataFrame
print("roger3")
data = pd.DataFrame.from_dict(dic_df)

# Drop duplicates in the first dataset based on the 'name' column
data.drop_duplicates(subset=['name'], inplace=True)

# Load the second dataset (fer2013.csv)
df = pd.read_csv(r"C:\Users\ASUS\Desktop\Music-recommendation-system\Spotify Dataset Analysis\data.csv")

# Drop duplicates in the second dataset based on the 'name' column
df.drop_duplicates(subset=['name'], inplace=True)

# Concatenate the two datasets
data1 = pd.concat([data, df], ignore_index=True)

# The 'name' column in the combined dataset
name = data1['name']

print("rogerA1")
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler

# col_features = ['danceability', 'energy', 'valence', 'loudness']
# X = MinMaxScaler().fit_transform(data1[col_features])
# kmeans = KMeans(init="k-means++",n_clusters=2,random_state=15).fit(X)
# data1['kmeans'] = kmeans.labels_
# # print(silhouette_score(X, data1['kmeans'], metric='euclidean'))

# data2 = data1[:data.shape[0]]
# cluster = data2.groupby(by='kmeans')  # Directly use the column name
# data2.pop('kmeans')
# print(data2,"hyper")
# #print(cluster)
# df1 = data2.sort_values(lambda x :x.sort_values(by="popularity"))

# df1.reset_index(level=0, inplace=True) 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Specify feature columns for clustering
col_features = ['danceability', 'energy', 'valence', 'loudness']

# Ensure feature columns exist and contain no NaN values
if not all(col in data1.columns for col in col_features):
    raise ValueError(f"Missing required columns in data1: {col_features}")

# Drop rows with missing values in the selected columns
data1_cleaned = data1.dropna(subset=col_features)

# Scale the features
X = MinMaxScaler().fit_transform(data1_cleaned[col_features])

# Apply KMeans clustering
try:
    kmeans = KMeans(init="k-means++", n_clusters=2, random_state=15).fit(X)
except ValueError as e:
    print(f"Error during KMeans fitting: {e}")
    raise

# Assign cluster labels to the data
data1_cleaned['kmeans'] = kmeans.labels_

# Separate the cleaned data into two groups based on the original data's shape
data2 = data1_cleaned.iloc[:data.shape[0]].copy()

# Group by the 'kmeans' column for cluster analysis
cluster = data2.groupby(by='kmeans')  # Grouped by clusters

# Remove the 'kmeans' column from data2 to avoid issues in further processing
data2 = data2.drop(columns=['kmeans'])

# Print grouped data
print(data2, "hyper")

# Sort values by the 'popularity' column if it exists
if 'popularity' in data2.columns:
    data2 = data2.sort_values(by='popularity')
else:
    raise ValueError("'popularity' column is missing in the dataset.")

# Reset index for the sorted data
data2.reset_index(drop=True, inplace=True)

# Assign the sorted DataFrame to df1
df1 = data2

# Final DataFrame print for debugging
print(df1)


def get_results(emotion_code):
    NUM_RECOMMEND = 10
    happy_set = []
    sad_set = []
    if emotion_code == 0:
        happy_set.append(df1[df1['kmeans'] == 0]['name'].head(NUM_RECOMMEND))
        return pd.DataFrame(happy_set).T
    else:
        sad_set.append(df1[df1['kmeans'] == 1]['name'].head(NUM_RECOMMEND))
        return pd.DataFrame(sad_set).T

print("roger4")
# def final():
#     root1 = tk.Tk()
#     root1.title("Your Playlist")
#     root1.configure(bg='black')

#     df = get_results(emotion_code)
#     cols = list(df.columns)
#     tree = ttk.Treeview(root1)
#     tree.pack(side=tk.TOP, fill=tk.X)
#     tree["columns"] = cols
#     for k in cols:
#         tree.column(k, anchor="w")
#         tree.heading(k, text=k, anchor='w')

#     for index, row in df.iterrows():
#         tree.insert("", 0, text=index, values=list(row))

#     root1.mainloop()
#     if emotion_word == 'sad':
#         print('emotion detected is SAD')
#     else:
#         print('emotion detected is HAPPY')


# emotion_word = (emotion_testing())
# if emotion_word == 'sad':
#     emotion_code = 0
# else:
#     emotion_code = 1

# window = tk.Tk()
# window.title("Music Recommender System")
# window.configure(background='black')
# window.grid_rowconfigure(0, weight=1)
# window.grid_columnconfigure(0, weight=1)
# message = tk.Label(
#     window, text="Music Recommender System",
#     bg="yellow", fg="black", width=50,
#     height=3, font=('times', 30, 'bold'))

# message.place(x=200, y=20)
# pred = tk.Button(window, text="print",
#                  command=final, fg="white", bg="black",
#                  width=20, height=3, activebackground="Red",
#                  font=('times', 15, ' bold '))
# pred.place(x=200, y=500)

# quitWindow = tk.Button(window, text="Quit",
#                        command=window.destroy, fg="white", bg="black",
#                        width=20, height=3, activebackground="Red",
#                        font=('times', 15, ' bold '))
# quitWindow.place(x=1100, y=500)

# image1 = Image.open("musicimg (1).jpg")
# test = ImageTk.PhotoImage(image1)

# label1 = tk.Label(image=test)
# label1.image = test
# label1.place(x=470, y=150)
# root.mainloop()
# window.mainloop()
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import time

# Dummy function for emotion testing
def emotion_testing():
    # Simulate an emotion detection process
    time.sleep(1)  # Simulate delay
    return 'happy'  # or 'sad'

def get_results(emotion_code):
    # Dummy function to simulate the return of a dataframe based on emotion code
    import pandas as pd
    data = {
        "song_name": ["Song1", "Song2", "Song3"],
        "popularity": [50, 70, 80],
    }
    return pd.DataFrame(data)

def final():
    # Capture the image from the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        return

    # Save the captured image (if needed)
    cv2.imwrite("captured_image.jpg", frame)

    # Display the captured image in the OpenCV window
    cv2.imshow('Captured Image', frame)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the OpenCV window
    
    # Release the camera resources
    cap.release()

    # Create and show the playlist window
    root1 = tk.Tk()
    root1.title("Your Playlist")
    root1.configure(bg='black')

    df = get_results(emotion_code)
    cols = list(df.columns)
    tree = ttk.Treeview(root1)
    tree.pack(side=tk.TOP, fill=tk.X)
    tree["columns"] = cols
    for k in cols:
        tree.column(k, anchor="w")
        tree.heading(k, text=k, anchor='w')

    for index, row in df.iterrows():
        tree.insert("", 0, text=index, values=list(row))

    root1.mainloop()

    # Emotion output
    if emotion_word == 'sad':
        print('Emotion detected is SAD')
    else:
        print('Emotion detected is HAPPY')

# Emotion word detection
emotion_word = emotion_testing()
if emotion_word == 'sad':
    emotion_code = 0
else:
    emotion_code = 1

# Main window
window = tk.Tk()
window.title("Music Recommender System")
window.configure(background='black')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(
    window, text="Music Recommender System",
    bg="yellow", fg="black", width=50,
    height=3, font=('times', 30, 'bold'))

message.place(x=200, y=20)
pred = tk.Button(window, text="Show Playlist",
                 command=final, fg="white", bg="black",
                 width=20, height=3, activebackground="Red",
                 font=('times', 15, ' bold '))
pred.place(x=200, y=500)

quitWindow = tk.Button(window, text="Quit",
                       command=window.destroy, fg="white", bg="black",
                       width=20, height=3, activebackground="Red",
                       font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)

# Display image on the Tkinter window
image1 = Image.open(r"C:\Users\ASUS\Desktop\Music-recommendation-system\images\musicimg (1).jpg")
test = ImageTk.PhotoImage(image1)

label1 = tk.Label(image=test)
label1.image = test
label1.place(x=470, y=150)

window.mainloop()

print("roger6")
# import time
# import numpy as np
# import pandas as pd
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
# from tkinter import ttk
# from tkinter import messagebox
# import tkinter as tk
# from PIL import Image, ImageTk
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import KMeans
# from emotion_video_classifier import emotion_testing

# # Set up Spotify client
# cid = "e45c35b36db446a0a4baffdc99b1272e"
# secret = "a1ff2bb04f9d4beaaf41f8f504b3ba04"
# client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
# sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# # Set up the Tkinter root window for entering artist name
# root = tk.Tk()
# root.title('CREDENTIALS')
# root.geometry("600x400")
# root.configure(bg='black')

# name1 = tk.StringVar()
# image_path = r"C:\Users\ASUS\Desktop\Music-recommendation-system\images\musicback.jpg"
# file = Image.open(image_path)
# photo = ImageTk.PhotoImage(file)
# l = tk.Label(root, image=photo)
# l.image = photo
# l.grid()

# def submit():
#     global name
#     name = name_entry.get()
#     messagebox.showinfo("Information", "Wait for sometime for us to create Playlists")
#     root.destroy()

# name_label = tk.Label(root, text='Enter Name of Artist', font=('calibre', 10, 'bold'))
# name_entry = tk.Entry(root, textvariable=name1, font=('calibre', 10, 'normal'))
# sub_btn = tk.Button(root, text='Submit', command=submit)

# name_label.grid(row=0, column=0)
# name_entry.grid(row=3, column=0)
# sub_btn.grid(row=5, column=0)
# root.mainloop()

# # Perform search and get albums for the artist
# result = sp.search(name)
# artist_uri = result['tracks']['items'][0]['artists'][0]['uri']
# sp_albums = sp.artist_albums(artist_uri, album_type='album')
# album_names = [album['name'] for album in sp_albums['items']]
# album_uris = [album['uri'] for album in sp_albums['items']]

# # Fetch album tracks
# def albumSongs(uri):
#     album = uri
#     spotify_albums[album] = {'album': [], 'track_number': [], 'id': [], 'name': [], 'uri': []}
#     tracks = sp.album_tracks(album)
#     for n in range(len(tracks['items'])):
#         spotify_albums[album]['album'].append(album_names[album_count])
#         spotify_albums[album]['track_number'].append(tracks['items'][n]['track_number'])
#         spotify_albums[album]['id'].append(tracks['items'][n]['id'])
#         spotify_albums[album]['name'].append(tracks['items'][n]['name'])
#         spotify_albums[album]['uri'].append(tracks['items'][n]['uri'])

# spotify_albums = {}
# album_count = 0
# for i in album_uris:
#     albumSongs(i)
#     album_count += 1

# # Get audio features for each track
# def audio_features(album):
#     spotify_albums[album].update({
#         'acousticness': [], 'danceability': [], 'energy': [], 'instrumentalness': [], 'liveness': [],
#         'loudness': [], 'speechiness': [], 'tempo': [], 'valence': [], 'popularity': []
#     })
#     for track in spotify_albums[album]['uri']:
#         try:
#             features = sp.audio_features(track)
#             if features[0]:
#                 spotify_albums[album]['acousticness'].append(features[0]['acousticness'])
#                 spotify_albums[album]['danceability'].append(features[0]['danceability'])
#                 spotify_albums[album]['energy'].append(features[0]['energy'])
#                 spotify_albums[album]['instrumentalness'].append(features[0]['instrumentalness'])
#                 spotify_albums[album]['liveness'].append(features[0]['liveness'])
#                 spotify_albums[album]['loudness'].append(features[0]['loudness'])
#                 spotify_albums[album]['speechiness'].append(features[0]['speechiness'])
#                 spotify_albums[album]['tempo'].append(features[0]['tempo'])
#                 spotify_albums[album]['valence'].append(features[0]['valence'])
#                 pop = sp.track(track)
#                 spotify_albums[album]['popularity'].append(pop['popularity'])
#         except Exception as e:
#             print(f"Error fetching audio features for track {track}: {e}")

# # Fetch audio features for all albums
# for i in spotify_albums:
#     audio_features(i)
#     time.sleep(np.random.uniform(2, 5))

# # Prepare data for clustering
# dic_df = {'album': [], 'track_number': [], 'id': [], 'name': [], 'uri': [], 'acousticness': [], 'danceability': [],
#           'energy': [], 'instrumentalness': [], 'liveness': [], 'loudness': [], 'speechiness': [], 'tempo': [],
#           'valence': [], 'popularity': []}

# for album in spotify_albums:
#     for feature in spotify_albums[album]:
#         dic_df[feature].extend(spotify_albums[album][feature])

# data = pd.DataFrame.from_dict(dic_df)
# data.drop_duplicates(inplace=True, subset=['name'])

# # Combine data with external dataset
# df = pd.read_csv(r"C:\Users\ASUS\Desktop\Music-recommendation-system\Spotify Dataset Analysis\fer2013.csv")
# df.drop_duplicates(inplace=True, subset=['name'])
# data1 = pd.concat([data, df], ignore_index=True)

# # Clustering
# col_features = ['danceability', 'energy', 'valence', 'loudness']
# X = MinMaxScaler().fit_transform(data1[col_features])
# kmeans = KMeans(init="k-means++", n_clusters=2, random_state=15).fit(X)
# data1['kmeans'] = kmeans.labels_

# # Sort and group the data
# data2 = data1[:data.shape[0]]
# data2.pop('kmeans')
# df1 = data2.sort_values(by="popularity", ascending=False)

# # Get recommendations based on emotion
# def get_results(emotion_code):
#     NUM_RECOMMEND = 10
#     if emotion_code == 0:
#         return df1[df1['kmeans'] == 0]['name'].head(NUM_RECOMMEND)
#     else:
#         return df1[df1['kmeans'] == 1]['name'].head(NUM_RECOMMEND)

# # Display final playlist
# def final():
#     root1 = tk.Tk()
#     root1.title("Your Playlist")
#     root1.configure(bg='black')

#     df = get_results(emotion_code)
#     tree = ttk.Treeview(root1)
#     tree.pack(side=tk.TOP, fill=tk.X)
#     tree["columns"] = ["name"]
#     tree.heading("#0", text="Index")
#     tree.heading("name", text="Track Name")

#     for index, row in df.iterrows():
#         tree.insert("", 0, text=index, values=(row["name"],))

#     root1.mainloop()

# # Determine the emotion and generate the playlist
# emotion_word = emotion_testing()
# emotion_code = 0 if emotion_word == 'sad' else 1

# window = tk.Tk()
# window.title("Music Recommender System")
# window.configure(background='black')
# window.grid_rowconfigure(0, weight=1)
# window.grid_columnconfigure(0, weight=1)

# message = tk.Label(window, text="Music Recommender System", bg="yellow", fg="black", width=50, height=3,
#                    font=('times', 30, 'bold'))
# message.place(x=200, y=20)

# pred = tk.Button(window, text="Generate Playlist", command=final, fg="white", bg="black", width=20, height=3,
#                  activebackground="Red", font=('times', 15, 'bold'))
# pred.place(x=200, y=500)

# quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="white", bg="black", width=20, height=3,
#                        activebackground="Red", font=('times', 15, 'bold'))
# quitWindow.place(x=1100, y=500)

# window.mainloop()
