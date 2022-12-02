# imports and setup
import lyricsgenius
import os
from datetime import datetime

## ========================================== FUNCTIONS ==========================================

# function to save every song to one file
def writeSongToFile(artist):
    for song in artist.songs:
        # save song in a file called artistname-songname.txt
        song.save_lyrics(filename='Lyrics_' + song.artist + '_' + song.title, extension='txt', verbose=True)

# function to move all files starting with 'Lyrics_' to path
def moveLyricsFiles(fromPath, toPath):
    for filename in os.listdir(fromPath):
        if filename.startswith('Lyrics_'):
            os.rename(filename, toPath + filename)

## ========================================== MAIN ==========================================


root = './'

# API token path
tokenPath = root + 'geniusToken.txt'

# create logfile 
logPath = root + 'log.txt'
logFile = open(logPath, 'w')

# init logfile with timestamp
logFile.write('LOG STARTED', str(datetime.datetime.now()))



# Genius API Access Token
# get token at: https://genius.com/api-clients
# load token from file
with open(tokenPath, 'r') as file:
    GENIUS_ACCESS_TOKEN = str(file.read())

    # delete last newline character from token
    # GENIUS_ACCESS_TOKEN = GENIUS_ACCESS_TOKEN[:-1]
    file.close()

print(GENIUS_ACCESS_TOKEN)

# init crawler with token'
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)



# try 10 times for every artist
for i in range(10):
    try:
        # get artist
        artist = genius.search_artist('The Weeknd', max_songs=100, sort='title')
        # save every song to one file
        writeSongToFile(artist)
        # move all files to root
        moveLyricsFiles(root, root)
        break
    except:
        print('Error occured, trying again...')



# for manual data generation
artistFile = genius.search_artist("JAY-Z", max_songs=50, sort="popularity")


writeSongToFile(artistFile)
moveLyricsFiles(root, "./datasets/add/")