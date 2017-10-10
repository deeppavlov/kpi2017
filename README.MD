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

Run model training
```sh
pyb train_<model_name>
```
where <model_name> should be replaced with the name of the model from the library: ner, paraphraser, insults, squad.

To test all the models use
```sh
pyb run_unit_tests -v
```
After that, detailed report could be found in target/reports.

Works on Ubuntu 16.04.
