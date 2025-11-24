# Base de Conhecimento — Chatbot de Suporte

## Autenticação de API
- Códigos 401 indicam problemas de autenticação.
- Verifique se a chave de API está ativa e com permissões adequadas.
- Respeite limites de taxa e implemente retries exponenciais.

## Checklist de Deploy
1. Executar testes unitários e de integração.
2. Validar variáveis de ambiente e secrets.
3. Configurar monitoramento e alertas.
4. Planejar rollback e comunicar janelas de manutenção.

## Triagem de Tickets
- Prioridade alta para incidentes que afetam produção.
- Títulos claros (até 140 caracteres) e objetivos.
- Use tags e categorias para facilitar a triagem.

## Logs e Observabilidade
- Registre contexto: usuário, endpoint, correlação.
- Centralize logs e mantenha retenção adequada.
- Use tracing para identificar gargalos.

## Boas Práticas de Suporte
- Confirme o ambiente (produção, staging, dev).
- Reproduza o problema com passos mínimos.
- Documente a solução e atualize a base de conhecimento.