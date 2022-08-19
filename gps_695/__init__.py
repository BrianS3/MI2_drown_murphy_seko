print("gps_695 package load")
import pip
import json

try:
    import dotenv
except:
    pip.main(['install', 'python-dotenv'])

try:
    import requests
except:
    pip.main(['install', 'requests'])
