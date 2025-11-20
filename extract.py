import zipfile


def main():
    print('extracting...')
    with zipfile.ZipFile('TransPath_data.zip', 'r') as zip_ref:
        zip_ref.extractall('./TransPath_data')
    print('done!')

if __name__ == '__main__':
    main()
