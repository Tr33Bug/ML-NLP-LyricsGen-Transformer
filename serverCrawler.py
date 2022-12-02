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
logFile.write('LOG STARTED')


# load list from file 
topArtists = []
with open('./datasets/ServerGen_top/topArtistsSave.txt', 'r') as f:
    for line in f:
        topArtists.append(line.strip())
topRapper = []
with open('./datasets/ServerGen_top/topRappersSave.txt', 'r') as f:
    for line in f:
        topRapper.append(line.strip())

print(topArtists[0])
print(topRapper[0])



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
for artistName in topArtists:
    for i in range(10):
        try:
            # get artist
            artist = genius.search_artist(artist, max_songs=100, sort='title')
            # save every song to one file
            writeSongToFile(artist)
            # move all files to root
            moveLyricsFiles(root, "./datasets/ServerGen_top/")
            logFile.write('Successfully crawled ' + artistName)
            break
        except:
            print('Error occured, trying again...')
            logFile.write('ERROR:', artistName, ' Error occured, trying again... file:', artist)

logFile.write('FINISHED CRAWLING TOPARTISTS\n')

# try 10 times for every artist
for artistName in topRapper:
    for i in range(10):
        try:
            # get artist
            artist = genius.search_artist(artistName, max_songs=100, sort='title')
            # save every song to one file
            writeSongToFile(artist)
            # move all files to root
            moveLyricsFiles(root, "./datasets/ServerGen_rap/")
            logFile.write('Successfully crawled ' + artistName)
            break
        except:
            print('Error occured, trying again...')
            logFile.write('ERROR:', artistName, ' Error occured, trying again... file:', artist)

logFile.write('FINISHED CRAWLING RAPARTISTS\n')

 
logFile.write('LOG ENDED')

logFile.close()