import os
import sys
import uuid
import importlib
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1.Agent"))

build_agent_graph = importlib.import_module("4_graph_builder").build_agent_graph
agent_app = build_agent_graph()

conversations = [
    (
        'ENGLISH',
        [
            'Hello, I want to search for a 7 million villa in New Cairo with 4 bedrooms.',
            'Please give me the best available match and explain why.'
        ],
        'en'
    ),
    (
        'ARABIC',
        [
            'عايز فيلا 4 غرف في التجمع الخامس بحد أقصى 15 مليون',
            'ممكن تشرحلي أفضل اختيار وليه هو الأفضل؟'
        ],
        'ar-EG'
    )
]

for label, turns, lang in conversations:
    print('===', label, 'conversation ===')
    thread_id = str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    state = {
        'current_filters': {},
        'tool_outputs': {},
        'missing_info': [],
        'active_intent': 'idle',
        'booking_details': {},
        'confidence_score': 1.0,
        'is_out_of_domain': False,
        'audit_retries': 0,
        'user_language': lang,
        'recent_properties': [],
        'discussion_context': {},
        'dialogue_state': {},
        'reference_map': {},
        'focus_listing_id': None,
        'last_recommendation_snapshot': [],
        'response_mode': 'tool_required',
        'dialogue_act': 'general',
        'target_property_refs': [],
        'carry_forward_slots': {},
        'slot_updates': {},
        'response_plan': {},
    }
    for idx, text in enumerate(turns, start=1):
        state['messages'] = [HumanMessage(content=text)]
        result = agent_app.invoke(state, config)
        messages = result.get('messages', [])
        print(f'Turn {idx}:', text)
        print('Agent:', messages[-1].content if messages else '<no response>')
        print()
