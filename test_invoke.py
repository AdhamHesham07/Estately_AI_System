import uuid, importlib, sys, os
sys.path.append(os.path.abspath('1.Agent'))
print("Importing LangChain...", flush=True)
from langchain_core.messages import HumanMessage
print("Importing Graph Builder...", flush=True)
build_graph = importlib.import_module('4_graph_builder').build_agent_graph
print("Building Agent...", flush=True)
agent_app = build_graph()
config={'configurable': {'thread_id': str(uuid.uuid4())}}
inputs={'messages': [HumanMessage(content='hi')], 'user_language': 'en'}
print("Invoking...", flush=True)
import traceback
try:
    result = agent_app.invoke(inputs, config)
    print("RESULT:", result, flush=True)
except Exception as e:
    traceback.print_exc()
print('Done!', flush=True)
