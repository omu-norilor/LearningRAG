
- cd code
- docker compose -f ../infra/docker-compose.yaml build 
- docker compose -f ../infra/docker-compose.yaml up -d
- docker exec -it rag-dev bash
- docker exec -it ollama-service ollama pull llama3.2:3b
-docker
