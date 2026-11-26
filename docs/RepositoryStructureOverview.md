linguameet-ai/
│
├── backend/
│   ├── README.md
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── pytest.ini
│   ├── .env.example
│   │
│   ├── api/
│   │   ├── README.md
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   ├── middleware/
│   │   │   ├── README.md
│   │   │   ├── error_handler.py
│   │   │   ├── request_logging.py
│   │   │   └── tests/
│   │   │       └── test_middleware.py
│   │   │
│   │   ├── routes/
│   │   │   ├── README.md
│   │   │   ├── asr_routes.py
│   │   │   ├── translation_routes.py
│   │   │   ├── tts_routes.py
│   │   │   ├── meeting_routes.py
│   │   │   ├── health_routes.py
│   │   │   └── tests/
│   │   │       ├── README.md
│   │   │       ├── test_asr_routes.py
│   │   │        ├── test_translation_routes.py
│   │   │        ├── test_tts_routes.py
│   │   │        ├── test_meeting_routes.py
│   │   │        └── test_health_routes.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── README.md
│   │   │   ├── asr_schema.py
│   │   │   ├── translation_schema.py
│   │   │   ├── tts_schema.py
│   │   │   ├── meeting_schema.py
│   │   │   └── tests/
│   │   │       └── test_schemas.py
│   │   │
│   │   └── tests/
│   │       ├── README.md
│   │       ├── conftest.py
│   │       ├── test_main.py
│   │       └── test_api_integration.py
│   │
│   ├── models/
│   │   ├── README.md
│   │   │
│   │   ├── asr/
│   │   │   ├── README.md
│   │   │   ├── asr_model.py
│   │     ├── asr_config.py
│   │   │   ├── model_loader.py
│   │     ├── requirements.txt
│   │     └── tests/
│   │   │       ├── README.md
│   │   │       ├── test_asr_model.py
│   │   │        └── test_model_loader.py
│   │   │
│   │   ├── translation/
│   │   │   ├── README.md
│   │   │   ├── translation_model.py
│   │     ├── language_mapping.py
│   │     ├── translation_config.py
│   │     ├── requirements.txt
│   │     └── tests/
│   │             ├── README.md
│   │     ├── test_translation_model.py
│   │     └── test_language_mapping.py
│   │   │
│   │   ├── tts/
│   │   │   ├── README.md
│   │     ├── tts_model.py
│   │     ├── voice_manager.py
│   │     ├── tts_config.py
│   │     ├── requirements.txt
│   │     └── tests/
│   │             ├── README.md
│   │     ├── test_tts_model.py
│   │     └── test_voice_manager.py
│   │   │
│   │   └── shared/
│   │   │   ├── README.md
│   │     ├── model_utils.py
│   │     ├── device_manager.py
│   │     ├── audio_utils.py
│   │     └── tests/
│   │             └── test_model_utils.py
│   │
│   ├── services/
│   │   ├── README.md
│   │   │
│   │   ├── vad/
│   │   │   ├── README.md
│   │     ├── vad_service.py
│   │     ├── vad_config.py
│   │     └── tests/
│   │             └── test_vad_service.py
│   │   │
│   │   ├── diarization/
│   │   │   ├── README.md
│   │     ├── diarization_service.py
│   │     ├── diarization_config.py
│   │     └── tests/
│   │             └── test_diarization_service.py
│   │   │
│   │   ├── language_routing/
│   │   │   ├── README.md
│   │     ├── routing_service.py
│   │     ├── routing_config.py
│   │     └── tests/
│   │             └── test_routing_service.py
│   │   │
│   │   ├── audio_processing/
│   │   │   ├── README.md
│   │     ├── audio_normalizer.py
│   │     ├── audio_converter.py
│   │     ├── audio_buffer.py
│   │     └── tests/
│   │             └── test_audio_processing.py
│   │   │
│   │   └── meeting/
│   │   │   ├── README.md
│   │     ├── session_service.py
│   │     ├── participant_service.py
│   │     ├── subtitle_service.py
│   │     └── tests/
│   │               ├── test_session_service.py
│   │     ├── test_participant_service.py
│   │     └── test_subtitle_service.py
│   │
│   ├── realtime/
│   │   ├── README.md
│   │   ├── livekit_agent.py
│   │   ├── websocket_manager.py
│   │   ├── stream_processor.py
│   │   ├── latency_monitor.py
│   │   ├── event_manager.py
│   │   ├── tests/
│   │   │   ├── README.md
│   │     ├── test_livekit_agent.py
│   │     ├── test_websocket_manager.py
│   │     ├── test_stream_processor.py
│   │     └── test_latency_monitor.py
│   │
│   ├── infrastructure/
│   │   ├── README.md
│   │   │
│   │   ├── database/
│   │   │   ├── README.md
│   │     ├── connection.py
│   │     ├── models.py
│   │     ├── migrations/
│   │     └── tests/
│   │     │   └── test_database.py
│   │     │
│   │   ├── cache/
│   │   │   ├── README.md
│   │     ├── redis_client.py
│   │     ├── cache_service.py
│   │     └── tests/
│   │     │   └── test_cache.py
│   │     │
│   │   └── storage/
│   │         ├── README.md
│   │     ├── audio_storage.py
│   │     ├── file_storage.py
│   │     └── tests/
│   │     │   └── test_storage.py
│   │
│   ├── config/
│   │   ├── README.md
│   │   ├── settings.yaml
│   │   ├── logging.yaml
│   │   ├── model_settings.yaml
│   │   └── tests/
│   │     └── test_settings.py
│   │
│   ├── deployment/
│   │   ├── README.md
│   │   ├── Dockerfile
│   │   ├── Dockerfile.gpu
│   │   ├── docker-compose.yml
│   │   ├── docker-compose.gpu.yml
│   │   ├── .dockerignore
│   │   │
│   │   ├── kubernetes/
│   │   │   ├── README.md
│   │     ├── api-deployment.yaml
│   │     ├── redis-deployment.yaml
│   │     ├── gpu-node-config.yaml
│   │     └── tests/
│   │     │   └── test_manifests.py
│   │   │
│   │   └── scripts/
│   │     ├── README.md
│   │     ├── start_local.sh
│   │     ├── start_gpu.sh
│   │     ├── stop_services.sh
│   │     ├── deploy_cloud.sh
│   │     └── tests/
│   │     │   └── test_deployment_scripts.sh
│   │
│   ├── tests/
│   │   ├── README.md
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── contract/
│   │   ├── fixtures/
│   │   └── conftest.py
│   │
│   └── README.md
│
├── frontend/
│   ├── README.md
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.json
│     ├── vite.config.ts
│     ├── vitest.config.ts
│     ├── eslint.config.js
│     ├── .env.example
│     │
│     ├── src/
│     │   ├── README.md
│     │   ├── main.tsx
│       ├── App.tsx
│     │   │
│     │   ├── components/
│       │   │   ├── README.md
│       │     │
│     │     ├── video/
│         │   │   ├── README.md
│           │   ├── VideoTile.tsx
│             └── tests/
│                   └── VideoTile.test.tsx
│             │
│             ├── subtitles/
│               ├── README.md
│               ├── SubtitleOverlay.tsx
│               └── tests/
│                     └── SubtitleOverlay.test.tsx
│               │
│               ├── language/
│                 ├── README.md
│                 ├── LanguageSelector.tsx
│                 └── tests/
│                     └── LanguageSelector.test.tsx
│                 │
│               └── meeting/
│                     ├── README.md
│                 ├── MeetingControls.tsx
│                 └── tests/
│                       └── MeetingControls.test.tsx
│            
│          ├── pages/
│          │   ├── README.md
│          │   ├── HomePage.tsx
│          │   ├── MeetingPage.tsx
│          │   ├── SettingsPage.tsx
│          └── tests/
│                 ├── HomePage.test.tsx
│                 ├── MeetingPage.test.tsx
│                 └── SettingsPage.test.tsx
│        
│        ├── hooks/
│        │   ├── README.md
│        │   ├── useWebRTC.ts
│        │   ├── useTranslation.ts
│        │   ├── useSubtitles.ts
│        │   └── tests/
│                 ├── useWebRTC.test.ts
│                 ├── useTranslation.test.ts
│                 └── useSubtitles.test.ts
│        
│        ├── services/
│        │   ├── README.md
│        │   ├── apiClient.ts
│        │   ├── websocketClient.ts
│        │   └── tests/
│                 ├── apiClient.test.ts
│               └── websocketClient.test.ts
│        
│        ├── stores/
│        │   ├── README.md
│        │   ├── meetingStore.ts
│        │   ├── settingsStore.ts
│        │   └── tests/
│                 ├── meetingStore.test.ts
│               └── settingsStore.test.ts
│        
│        ├── utils/
│        │   ├── README.md
│        │   ├── audioUtils.ts
│        │   ├── languageUtils.ts
│        │   └── tests/
│                 ├── audioUtils.test.ts
│               └── languageUtils.test.ts
│        
│        ├── types/
│        │   ├── README.md
│        │   ├── api.ts
│        │  ├── meeting.ts
│        │  └── language.ts
│        
│        ├── config/
│        │   ├── README.md
│        │  ├── env.ts
│        │  └── tests/
│                 └── env.test.ts
│        
│        └── styles/
│            ├── README.md
│            └── tailwind.css
│      
│      ├── public/
│      │   ├── README.md
│      │  └── icons/
│      
│    ├── tests/
│    │   ├── README.md
│    │   ├── setup.ts
│    │  ├── integration/
│    │  ├── accessibility/
│    │  └── e2e/
│    
│    └── README.md
│
├── realtime/
│   ├── README.md
│   ├── architecture.md
│   ├── protocol.md
│   ├── events/
│   │   ├── README.md
│   │   ├── event_types.py
│     │   └── tests/
│    │       └── test_event_types.py
│    └── tests/
│       ├── README.md
│       └── test_realtime_integration.py
│
├── postman/
│   ├── README.md
│   ├── collections/
│   │   ├── LinguaMeet-API.postman_collection.json
│     │   └── LinguaMeet-Realtime.postman_collection.json
│    ├── environments/
│    │   ├── local.postman_environment.json
│    │   └── docker.postman_environment.json
│    └── tests/
│       └── README.md
│
├── docs/
│   ├── README.md
│   ├── architecture/
│   │   ├── README.md
│    │   ├── architecture-diagram.png
│    │  ├── system-overview.md
│    │  ├── data-flow.md
│    │  └── adr/
│    │       └── README.md
│    │
│    ├── api/
│    │   ├── README.md
│    │  ├── api-reference.md
│    │  ├── authentication.md
│    │  └── error-codes.md
│    │
│    ├── deployment/
│    │   ├── README.md
│    │  ├── local-development.md
│    │  ├── docker-guide.md
│    │  ├── gpu-setup.md
│    │  └── cloud-deployment.md
│    │
│    └── development/
│        ├── README.md
│        ├── coding-standards.md
│        ├── testing-guide.md
│        └── contributing.md
│
├── tests/
│   ├── README.md
│   ├── integration/
│   ├── performance/
│   ├── load/
│   └── e2e/
│
├── scripts/
│   ├── README.md
│   ├── setup_env.sh
│   ├── run_backend_tests.sh
│   ├── run_frontend_tests.sh
│   ├── run_all_tests.sh
│   ├── benchmark_latency.py
│   └── health_check.sh
│
├── .github/
│   └── workflows/
│       ├── backend-ci.yml
│       ├── frontend-ci.yml
│       └── integration-ci.yml
│
├── .env.example
├── .gitignore
├── Makefile
├── LICENSE
└── README.md

