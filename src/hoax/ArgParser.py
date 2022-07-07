#!/usr/bin/env python
# coding: utf-8



import argparse



class ArgParser():
    def __init__(self):

        self.parser =  argparse.ArgumentParser()
        self.parser.add_argument("db", help="The database containg training data", type=str)
        self.parser.add_argument("config", help="The config file in json format", type=str)
        self.args = self.parser.parse_args()

    def getConfigName(self):
        return self.args.config

    def getDatabaseName(self):
        return self.args.db


