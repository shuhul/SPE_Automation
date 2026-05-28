import os
import plotter

DATA_FOLDER = 'data'

folders = [f for f in os.scandir(DATA_FOLDER) if f.is_dir()]
latest = max(folders, key=lambda f: f.stat().st_mtime)
foldername = latest.name
print(f'Opening: {foldername}')

result = plotter.select_emitters(foldername, 'coarse', data_folder=DATA_FOLDER)
print(f'Selected emitters: {result}')
