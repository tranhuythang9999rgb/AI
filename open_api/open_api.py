from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser


# Định nghĩa schema
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
index_dir = "indexdir"
index = create_in(index_dir, schema)

# Thêm tài liệu
writer = index.writer()
writer.add_document(title="First document", content="This is the content of the first document.")
writer.commit()

# Tìm kiếm
with index.searcher() as searcher:
    query = QueryParser("content", index.schema).parse("first")
    results = searcher.search(query)
    for result in results:
        print(result['title'])
