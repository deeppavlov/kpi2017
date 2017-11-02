"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os


class CoNLLClassificationMetrics(object):
    """Classification metrics class"""

    def __init__(self, model_files_path):
        """Initialization of metrics class

        Args:
            model_files_path: path to save temporary metrics files
        """
        self.model_files_path = model_files_path
        self.output_filename = 'ner_output.txt'
        self.report_filename = 'ner_report.txt'
        self.y_pred = []
        self.y_true = []

    def clear(self):
        """Clear all data from previous calls"""
        del self.y_pred[:]
        del self.y_true[:]

    def update(self, observation, y):
        """Observation accumulator

        Args:
            observation: observation data with predicted tags
            y: ground truth data
        """
        if y and 'text' in observation:
            y_true = y[0].split()
            y_pred = observation['text'].split()[:len(y_true)]
            self.y_true.append(y_true)
            self.y_pred.append(y_pred)

    def report(self):
        """Calculate and return metrics as a classification report"""
        if len(self.y_pred) > 0:
            output_file_path = os.path.join(self.model_files_path, self.output_filename)
            report_file_path = os.path.join(self.model_files_path, self.report_filename)
            if not os.path.isdir(self.model_files_path):
                os.mkdir(self.model_files_path)
            with open(output_file_path, 'w') as f:
                for tags_pred, tags_gt in zip(self.y_pred, self.y_true):
                    for tag_predicted, tag_ground_truth in zip(tags_pred, tags_gt):
                        f.write(' '.join(['pur'] * 5 + [tag_ground_truth] + [tag_predicted]) + '\n')
            conll_evaluation_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'conlleval')
            shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_file_path, report_file_path)
            os.system(shell_command)
            f1_score, accuracy = self.f_and_accuracy()
            report = {
                'f1': f1_score,
                'accuracy': accuracy,
                'cnt': len(self.y_pred)
            }
            return report
        return dict()

    def f_and_accuracy(self):
        """Calculate metrics"""
        report_filepath = os.path.join(self.model_files_path, self.report_filename)
        try:
            with open(report_filepath) as f:
                # Header to trash
                _ = f.readline()
                overall_items = [item for item in f.readline().split(' ') if len(item) > 0]
                f_score = float(overall_items[-1].strip())
                accuracy = float(overall_items[1][:-2])
            return f_score, accuracy
        except FileNotFoundError:
            raise FileNotFoundError('There is no ' + report_filepath +
                                    ' report file! conll script must be'
                                    ' run before extracting f-score')
