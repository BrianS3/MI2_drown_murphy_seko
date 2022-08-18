print("gps_695 package load")
import pip

try:
    __import__(dotenv)
except:
    pip.main(['install', 'python-dotenv'])

