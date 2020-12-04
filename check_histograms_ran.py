#!/usr/bin/env python
"""Run through folder to check uncorrupted outputs made where expected"""

from reame.validation import HistogramValidator

ids_to_rerun = []
for product, sensor in (
    ("ADV_v2.31", "aatsr"),
    ("ADV_v2.31", "atsr2"),
    ("SU_v4.3", "aatsr"),
    ("SU_v4.3", "atsr2"),
    ("ORAC_v4.01", "aatsr"),
    ("ORAC_v4.01", "atsr2"),
    ("ORAC_v4.01", "slstr-a"),
    ("ORAC_v4.01", "slstr-b"),
    ("APORAC_v4.01", "aatsr"),
    ("APORAC_v4.01", "atsr2"),
    ("APORAC_v4.01", "slstr-a"),
    ("APORAC_v4.01", "slstr-b"),
    ("DeepBlue_c61", "MOD"),
    ("DeepBlue_c61", "MYD"),
    #("DarkTarget_c61", "MOD"),
    #("DarkTarget_c61", "MYD"),
    #("MAIAC_c6", "MCD")
):
    ids = []
    validator = HistogramValidator(product, sensor)
    for test in validator:
        try:
            if not (test.check("output_valid") and test.check("file_list")):
                print(test)
                with open(test._fdr + f"/log/{product}_{sensor}_{test._offset}.err") as err_file:
                    for line in err_file.readlines():
                        if not line.startswith("cpu-bind"):
                            print("\t" + line)
                            ids.append(test._offset)
        except StopIteration:
            pass

    ids_to_rerun.append(ids)
