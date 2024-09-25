import grpc
from concurrent import futures
import time

import protos.pb_pb2 as pb__pb2
import protos.pb_pb2_grpc as pb_pb2_grpc

class AIService(pb_pb2_grpc.AIServiceServicer):
    def ProcessAIRequest(self, request, context):
        # Xử lý dữ liệu đầu vào và tạo output
        output_data = "Processed: " + request.input_data
        return pb__pb2.AIResponse(success=True, message="Request processed successfully", output_data=output_data)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_pb2_grpc.add_AIServiceServicer_to_server(AIService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server is running on port 50051...")
    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
