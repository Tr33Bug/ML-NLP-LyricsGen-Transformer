# print asciart of the word DatasetCreater by Overfitter
print('''
   _ \                       _| _)  |    |               
  |   | \ \   /  _ \   __|  |    |  __|  __|   _ \   __| 
  |   |  \ \ /   __/  |     __|  |  |    |     __/  |    
 \___/    \_/  \___| _|    _|   _| \__| \__| \___| _|    
                                                         ''')
print('''   ___|   ___| \ \     /        ___|                    |                
  |     \___ \  \ \   /        |       __|  _ \   _` |  __|   _ \    __| 
  |           |  \ \ / _____|  |      |     __/  (   |  |    (   |  |    
 \____| _____/    \_/         \____| _|   \___| \__,_| \__| \___/  _|    
                                                                         
''')


print('⏳⏳⏳ Starting CSV-Creator by Overfitter ⏳⏳⏳')

# imports and setup
import os
import sys
try:
    import pandas as pd
except ImportError:
    print('❌ --> Pandas is not installed, please install it with "pip install pandas"')
    sys.exit()




#### FILE PATHS
# root path
root = './'

#### FILE PATHS
geniusTopRapIMDBClean = root + 'datasets/geniusTopRapIMDBClean/'
geniusTopIMDBClean = root + 'datasets/geniusTopIMDBClean/'

exportPath = './datasets/df_songs.csv'
exportPathRap = './datasets/df_rap.csv'
exportPathTop = './datasets/df_top.csv'

print('\n')
print('Generating df_songs.csv from', geniusTopRapIMDBClean, 'and', geniusTopIMDBClean, '!')
print('----------------------------------------')


def createDfFromFiles(path):

    # delete '.' from path, if path is relative with '.' in beginning
    if path.startswith('./'):
        path = path[1:]
    
    # make the path from current directory and the input path
    pathComp = os.getcwd() + path

    # create a tuple from files in path dataset with the file name and the file content, with latin-1 encoding
    files = [(file, open(pathComp + file, 'r', encoding='latin-1').read()) for file in os.listdir(pathComp)]

    # delete all .txt from filename
    files = [(file.replace('.txt', ''), content) for file, content in files]

    # delete all Lyrics_ from filename
    files = [(file.replace('Lyrics_', ''), content) for file, content in files]

    # replace all double spaces with single space
    files = [(file, content.replace('  ', ' ')) for file, content in files]

    # delete all spaces at the beginning of the string
    files = [(file, content.lstrip()) for file, content in files]

    # create a dataframe from the tuple
    df = pd.DataFrame(files, columns=['Artist', 'Lyrics'])

    # split artist name into artist and song devided by _
    df[['Artist', 'Song']] = df.Artist.str.split("_", expand=True)

    # rearrange colums to artist, song, lyrics
    df = df[['Artist', 'Song', 'Lyrics']]
    
    print('✅ --> Creating dataframe from', path, 'done!')
    return df

#### CREATE DATAFRAMES

# cretae dataframe from top artists
df_top = createDfFromFiles(geniusTopIMDBClean)

# create dataframe from rap artists
df_rap = createDfFromFiles(geniusTopRapIMDBClean)

#### CLEAN DATA I

# print names of artists with less than 50 songs
less50 = []
# for each artist in df_top append artist name and number of songs to less50 if number of songs is less than 50
for artist in df_top['Artist'].unique():
    if len(df_top[df_top['Artist'] == artist]) < 50:
        less50.append((artist, len(df_top[df_top['Artist'] == artist])))

for artist in df_rap['Artist'].unique():
    if len(df_rap[df_rap['Artist'] == artist]) < 50:
        less50.append((artist, len(df_rap[df_rap['Artist'] == artist])))


# drop artists from less 50 from df_top and df_rap
for artist, songs in less50:
    df_top = df_top[df_top['Artist'] != artist]
    df_rap = df_rap[df_rap['Artist'] != artist]


# print number of artists in top and rap
print('    Number of Top Artists --> df_top:', len(df_top['Artist'].unique()))
print('    Number of Top Rappers --> df_rap:', len(df_rap['Artist'].unique()))

print('✅ --> Cleaning Dataframes I done!')

#### CREATE CSV
df_rap.to_csv(exportPath, index=False)
df_top.to_csv(exportPath, index=False)

print('✅ --> Exporting dataframes rap and top to', exportPath, 'done!\n')


#### MERGE DATAFRAMES

# merge datasets
df_songs = pd.concat([df_top, df_rap], ignore_index=True)
df_songs.head()

print('✅ --> Merging Dataframes I done!')
#### CLEAN DATA II

# check for empty entries in dataframes
print('    Empty entries in the dataframes:')
print('    df_top:', df_top.isnull().values.any())
print('    df_rap:', df_rap.isnull().values.any())


# add lyric word count
# append lyrics word count to df_songs
df_songs['LyricsWordCount'] = df_songs['Lyrics'].str.split().str.len()

# sort df_songs by lyrics word count
df_songs = df_songs.sort_values(by=['LyricsWordCount'], ascending=True)

df_songs = df_songs[df_songs['LyricsWordCount'] >= 80]


# drop artists with less than 30 songs
for artist in df_songs['Artist'].unique():
    if len(df_songs[df_songs['Artist'] == artist]) < 30:
        df_songs = df_songs[df_songs['Artist'] != artist]

print('✅ --> Cleaning Dataframe done!')
#### CREATE CSV

# export dataframe to csv to path
df_songs.to_csv(exportPath, index=False)

print('✅ --> Exporting dataframe to', exportPath, 'done!\n')