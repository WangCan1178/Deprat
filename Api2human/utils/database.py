import urllib
from pymongo import MongoClient
from bson.objectid import ObjectId
import streamlit as st

class DB(object):
    def __init__(self):
        user = "admin"
        password = "123456"
        host = '127.0.0.1'
        port = 27017
        client = MongoClient('mongodb://{0}:{1}@{2}:{3}'.format(urllib.parse.quote_plus(user),urllib.parse.quote_plus(password),host,port))
        db = client.apidata
        self.collection = db.api
        
    def findall(self):
        result = [i for i in self.collection.find()]
        return result
    
    def findbyid(self, apiid):
        result = [i for i in self.collection.find({"apiid":apiid})]
        return result
    
    def insert(self, apiid, apitype, apipath, apidesc, request, response):
        datajson = {
        "apiid": apiid, 
        "apitype": apitype, 
        "apipath": apipath, 
        "apidesc": apidesc, 
        "request": request, 
        "response": response
        }
        result  = self.collection.insert_one(datajson)
        print('insert', result)
        return result
    
    def delete(self, apiid):
        result  = self.collection.delete_one({"apiid": apiid})
        print('delete', result)
        return result
    
    def update(self, apiid, apitype, apipath, apidesc, request, response):
        datajson = {
        "apiid": apiid, 
        "apitype": apitype, 
        "apipath": apipath, 
        "apidesc": apidesc, 
        "request": request, 
        "response": response
        }
        result  = self.collection.update_one(datajson)
        print('update', result)
        return result
    
    def save(self, apiid, _apipath, _apitype, _apidesc, _request, _response):
        if (type(apiid).__name__!='int') or (type(_apitype).__name__!='int') or \
            (len(_apipath)==0) or (len(_apidesc)==0) or \
            (type(_apipath).__name__!='str') or (type(_apidesc).__name__!='str') or \
            (type(_request).__name__!='list') or (type(_response).__name__!='list'):
            return None
        datajson = {
        "apiid": apiid, 
        "apitype": _apitype, 
        "apipath": _apipath, 
        "apidesc": _apidesc,
        "request": _request, 
        "response": _response
        }
        # result = self.collection.update_one({"apiid":apiid}, datajson, upsert = True)
        result = self.collection.update_one({"apiid":apiid}, {"$set": datajson}, upsert = True)
        return result

# db = DB()

# datajson = {
#         "apiid":0,
#         "apitype":0,
#         "apipath":"uniform.www/activity",
#         "apidesc":"Retrieve a list of API Requests that have been made.",
#         "request":[
#         {
#         "name":"limit", "type":"integer",
#         "desc":"How many API Events should be retrieved in a single request.",
#         "required":0,
#         "format":None
#         }
#         ],
#         "response":[{
#         "name":"offset",
#         "type":"integer",
#         "desc":"How far into the collection of API Events should the response start",
#         "format":None
#         }
#         ]
#         }
# result = db.collection.update_one({"apiid":2}, {"$set": datajson}, upsert = True)
# print(result)