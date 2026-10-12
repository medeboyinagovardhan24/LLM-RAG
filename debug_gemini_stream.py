import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
print('start')
try:
    stream = client.models.generate_content_stream(model='gemini-2.0-flash', contents='Say hello in one word')
    print('stream type', type(stream))
    for i, chunk in enumerate(stream):
        print('chunk', i, 'type', type(chunk))
        print('has text', hasattr(chunk, 'text'), 'text', getattr(chunk, 'text', None))
        print('dir sample', [a for a in dir(chunk) if not a.startswith('_')][:20])
        if i >= 2:
            break
except Exception as e:
    import traceback
    traceback.print_exc()
print('done')
