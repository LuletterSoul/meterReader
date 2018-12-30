import os
import json
from unittest import TestCase

from util.JsonModifier import JsonModifier


class TestJsonModifier(TestCase):
    meter_id = "pressure2_11"
    config_dir = "../config"

    def testInitJsonModifier(self):
        json_modifier = JsonModifier(self.meter_id, self.config_dir)
        self.assertTrue(len(json_modifier.json_info.keys()) > 0)

    def testModifyConfigKv(self):
        json_modifier = JsonModifier(self.meter_id, self.config_dir)
        pair_1 = ("name", "testestestsetes")
        pair_2 = ("enableFit", True)
        json_modifier.modifyKv(pair_1[0], pair_1[1])
        json_modifier.modifyKv(pair_2[0], pair_2[1])
        config_file = open(json_modifier.src_config_path)
        json_info = json.load(config_file)
        self.assertEqual(json_info[pair_1[0]], pair_1[1])
        self.assertEqual(json_info[pair_2[0]], pair_2[1])
        config_file.close()

        pair_3 = ("narrme", "pressure2")
        pair_4 = ("enableFit", False)
        json_modifier.modifyKv(pair_3[0], pair_3[1])
        json_modifier.modifyKv(pair_4[0], pair_4[1])
        config_file = open(json_modifier.src_config_path)
        json_info = json.load(config_file)
        self.assertEqual(json_info[pair_3[0]], pair_3[1])
        self.assertEqual(json_info[pair_4[0]], pair_4[1])
        config_file.close()

    def testModifyConfigDic(self):
        json_modifier = JsonModifier(self.meter_id, self.config_dir)
        dic = {"name": "bababababab", "enableFit": True, "startPoint": {"x": 123, "y": 124}}
        json_modifier.modifyDic(dic)
        config_file = open(json_modifier.src_config_path)
        json_info = json.load(config_file)
        for key, value in dic.items():
            self.assertEqual(json_info[key], value)
        config_file.close()

    def testRevertChange(self):
        json_modifier = JsonModifier(self.meter_id, self.config_dir)
        dic = {"name": "bababababab", "enableFit": True, "startPoint": {"x": 123, "y": 124}}
        json_modifier.modifyDic(dic)
        json_modifier.revert(backward=2)
        backup_file = open(json_modifier.dump_target_path)
        config_file = open(json_modifier.src_config_path)
        json_info = json.load(config_file)
        backup_info = json.load(backup_file)
        for bkey, bvalue in backup_info.items():
            self.assertEqual(json_info[bkey], bvalue)
        backup_file.close()
        config_file.close()

    def testDELBackup(self):
        json_modifier = JsonModifier(self.meter_id, self.config_dir, revert_before_del=True)
        dic = {"name": "bababababab", "enableFit": True, "startPoint": {"x": 123, "y": 124}}
        json_modifier.modifyDic(dic)
        path = json_modifier.dump_target_path
        del json_modifier
        self.assertFalse(os.path.exists(path))
