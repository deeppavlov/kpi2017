To automatically get private datasets and pretrained models from ipavlov's ftp server,
please set the `IPAVLOV_FTP` environment variable:
```
export IPAVLOV_FTP=ftp://{user}:{password}@{host}
```
Install from internal PyPi server
```sh
pip install --extra-index-url http://{host}:{port}/ --trusted-host {host} deeppavlov
```

Install from git
```sh
git clone https://github.com/deepmipt/deeppavlov
cd deeppavlov
python setup.py develop
```

