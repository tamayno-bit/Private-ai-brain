"""Deterministic lossless UTF-8 byte tokenizer. Replaceable via explicit versioning."""
class ByteTokenizer:
    BOS=256;EOS=257;PAD=258;UNK=259
    version='utf8-byte-v1'
    vocab_size=260
    def encode(self,text:str,bos=False,eos=False):
        if not isinstance(text,str):raise TypeError('Expected string')
        return ([self.BOS] if bos else []) + list(text.encode('utf-8')) + ([self.EOS] if eos else [])
    def decode(self,ids):
        return bytes(i for i in ids if isinstance(i,int) and 0<=i<=255).decode('utf-8',errors='replace')
