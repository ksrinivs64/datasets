# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os

import datasets
import tarfile
import random

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
authors={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = 'CodeNetDataset'

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
file_URLs = {
    'codenetjava_classify': "java",
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class codenetjava(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="codenetjava_classify", version=VERSION, description="download the Java codenet classification dataset"),
    ]

    DEFAULT_CONFIG_NAME = "codenetjava_classify"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "codenetjava" or self.config.name == "codenetjava-small":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "code1": datasets.Value("string"),
                    "code2": datasets.Value("string"),
                    "label": datasets.Value("int"),
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        my_url = file_URLs[self.config.name] 
        data_dir = dl_manager.manual_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, my_url + '.train.tar'),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, my_url + '.test.tar'),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, my_url + '.dev.tar'),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        problems2files = {}
        problems_files2code = {}
        negative_sample_size = 100
        positive_sample_size = 100
        self.length = negative_sample_size + positive_sample_size
        with tarfile.open(filepath, "r") as tar:
            for tarinfo in tar:
                arr = tarinfo.name.split('/')

                if arr[4] not in problems2files:
                    problems2files[arr[4]] = []
                code = tar.extractfile(tarinfo).read()
                problems2files[arr[4]].append(tarinfo.name)
                problems_files2code[tarinfo.name] = str(code)

            # create a set of examples by simply sampling from each 'problem set' for positives and
            # negatives by pairing problems from different problem sets
            desired_positive_sample_size = positive_sample_size
            desired_negative_sample_size = negative_sample_size
            l = list(problems2files.keys())

            problem_pairs = set()
            positive_code_pairs = []
            negative_code_pairs = []

            for k in problems2files:
                problems = problems2files[k]
                chosen = 0
                sample_per_problem = int(desired_positive_sample_size / len(l))

                while chosen <= sample_per_problem:
                    problem_pair = random.sample(problems, 2)
                    problem_pair.sort()
                    key = '-'.join(problem_pair)
                    if key in problem_pairs:
                        continue
                    problem_pairs.add(key)
                    print('positive pair:' + key)
                    positive_code_pairs.append(
                        (problems_files2code[problem_pair[0]], problems_files2code[problem_pair[1]]))
                    chosen += 1
                chosen = 0

                sample_per_problem = int(desired_negative_sample_size / len(l))

                while chosen <= sample_per_problem:
                    problem_1 = random.sample(problems, 1)[0]
                    negative = k
                    while negative == k:
                        negative = random.randint(0, len(problems2files.keys()) - 1)

                    problems = problems2files[list(problems2files.keys())[negative]]
                    problem_2 = random.sample(problems, 1)[0]
                    problem_pair = [problem_1, problem_2]

                    key = '-'.join(problem_pair)
                    if key in problem_pairs:
                        continue
                    print('negative pair:' + key)
                    problem_pairs.add(key)
                    negative_code_pairs.append(
                        (problems_files2code[problem_pair[0]], problems_files2code[problem_pair[1]]))
                    chosen += 1
            all_pairs = []
            for i in positive_code_pairs:
                all_pairs.append((1, i))
            for i in negative_code_pairs:
                all_pairs.append((0, i))

            for i, example in all_pairs:
                yield i, {
                    "code1": example[0],
                    "code2": example[1]
                }

        
