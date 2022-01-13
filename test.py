
from lib.utils import load_embedding_model
import lib.globals as globals 

m,v = load_embedding_model()

print(m.vectors)

print(v[globals.PAD_TOKEN])