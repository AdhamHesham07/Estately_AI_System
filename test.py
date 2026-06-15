import uuid, importlib, sys, os
print("1")
sys.path.append(os.path.abspath('1.Agent'))
print("2")
from langchain_core.messages import HumanMessage
print("3")
build_graph = importlib.import_module('4_graph_builder').build_agent_graph
print("4")
agent_app = build_graph()
print("5")
config={'configurable': {'thread_id': str(uuid.uuid4())}}
inputs={'messages': [HumanMessage(content='hi')], 'user_language': 'en'}
print('Invoking...', flush=True)
import traceback
try:
    agent_app.invoke(inputs, config)
except Exception as e:
    traceback.print_exc()
print('Done!', flush=True)
