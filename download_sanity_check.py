import requests
from urllib.parse import urlencode
import zipfile


def main():
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/3D4mYvD8YXHuyA'

    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
    print('downloading...')
    download_response = requests.get(download_url)
    with open('Dataset_sanity_check.zip', 'wb') as f:
        f.write(download_response.content)
    print('extracting...')
    with zipfile.ZipFile('Dataset_sanity_check.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print('done!')

if __name__ == '__main__':
    main()
