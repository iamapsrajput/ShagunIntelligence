# Shagun Intelligence Architecture Diagrams

This document contains comprehensive architecture diagrams for the Shagun Intelligence platform using Mermaid syntax. These diagrams can be rendered in any Markdown viewer that supports Mermaid.

## Table of Contents

1. [System Overview](#system-overview)
2. [Multi-Agent Architecture](#multi-agent-architecture)
3. [Data Flow Diagram](#data-flow-diagram)
4. [Trading Workflow](#trading-workflow)
5. [Deployment Architecture](#deployment-architecture)
6. [Database Schema](#database-schema)
7. [API Architecture](#api-architecture)
8. [Security Architecture](#security-architecture)

## System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        A[React Dashboard]
        B[Mobile App]
        C[API Clients]
    end
    
    subgraph "API Gateway"
        D[Nginx/Load Balancer]
        E[Rate Limiter]
        F[Auth Service]
    end
    
    subgraph "Application Layer"
        G[FastAPI Backend]
        H[WebSocket Server]
        I[Background Workers]
    end
    
    subgraph "AI Agent Layer"
        J[Coordinator Agent]
        K[Market Analyst]
        L[Technical Indicator]
        M[Sentiment Analyst]
        N[Risk Manager]
        O[Trade Executor]
        P[Data Processor]
    end
    
    subgraph "Data Layer"
        Q[(PostgreSQL)]
        R[(Redis Cache)]
        S[Message Queue]
    end
    
    subgraph "External Services"
        T[Zerodha Kite API]
        U[OpenAI API]
        V[News APIs]
        W[Market Data Feed]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    G --> I
    G --> J
    H --> J
    J --> K
    J --> L
    J --> M
    J --> N
    J --> O
    J --> P
    K --> Q
    L --> Q
    M --> Q
    N --> Q
    O --> T
    P --> W
    G --> Q
    G --> R
    I --> S
    M --> V
    K --> U
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style J fill:#fff3e0
    style K fill:#f3e5f5
    style L fill:#f3e5f5
    style M fill:#f3e5f5
    style N fill:#f3e5f5
    style O fill:#f3e5f5
    style P fill:#f3e5f5
    style Q fill:#e8f5e9
    style R fill:#e8f5e9
    style T fill:#fff9c4
    style U fill:#fff9c4
```

## Multi-Agent Architecture

```mermaid
graph TD
    subgraph "Coordinator Agent"
        CA[Coordinator Agent<br/>Orchestration & Decision Fusion]
    end
    
    subgraph "Analysis Agents"
        MA[Market Analyst<br/>Trends & Patterns]
        TI[Technical Indicator<br/>Signals & Calculations]
        SA[Sentiment Analyst<br/>News & Social Media]
    end
    
    subgraph "Execution Agents"
        RM[Risk Manager<br/>Position Sizing & Limits]
        TE[Trade Executor<br/>Order Management]
        DP[Data Processor<br/>Real-time Processing]
    end
    
    subgraph "Shared Resources"
        SM[Shared Memory<br/>Agent States]
        MQ[Message Queue<br/>Event Bus]
        DB[(Database<br/>Persistent Storage)]
    end
    
    CA --> MA
    CA --> TI
    CA --> SA
    CA --> RM
    CA --> TE
    CA --> DP
    
    MA --> SM
    TI --> SM
    SA --> SM
    RM --> SM
    TE --> SM
    DP --> SM
    
    MA --> MQ
    TI --> MQ
    SA --> MQ
    RM --> MQ
    TE --> MQ
    DP --> MQ
    
    SM --> DB
    MQ --> DB
    
    CA -.->|Consensus<br/>Algorithm| RM
    RM -.->|Approved<br/>Trades| TE
    DP -.->|Market<br/>Data| MA
    DP -.->|Market<br/>Data| TI
    
    style CA fill:#ffeb3b
    style MA fill:#e3f2fd
    style TI fill:#e3f2fd
    style SA fill:#e3f2fd
    style RM fill:#ffebee
    style TE fill:#ffebee
    style DP fill:#ffebee
```

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Data Sources"
        MD[Market Data<br/>WebSocket]
        NF[News Feed<br/>REST API]
        SF[Social Feed<br/>REST API]
    end
    
    subgraph "Ingestion"
        DP[Data Processor]
        DV[Data Validator]
        DN[Data Normalizer]
    end
    
    subgraph "Processing"
        CE[Calculation Engine]
        ML[ML Models]
        AG[Aggregator]
    end
    
    subgraph "Storage"
        TS[(Time Series DB)]
        CH[(Cache Layer)]
        PG[(PostgreSQL)]
    end
    
    subgraph "Distribution"
        PS[Pub/Sub System]
        WS[WebSocket Server]
        AP[API Server]
    end
    
    MD --> DP
    NF --> DP
    SF --> DP
    DP --> DV
    DV --> DN
    DN --> CE
    DN --> ML
    CE --> AG
    ML --> AG
    AG --> TS
    AG --> CH
    AG --> PG
    TS --> PS
    CH --> PS
    PG --> PS
    PS --> WS
    PS --> AP
    
    style MD fill:#e1f5fe
    style NF fill:#e1f5fe
    style SF fill:#e1f5fe
    style TS fill:#e8f5e9
    style CH fill:#e8f5e9
    style PG fill:#e8f5e9
```

## Trading Workflow

```mermaid
sequenceDiagram
    participant U as User/Scheduler
    participant C as Coordinator
    participant MA as Market Analyst
    participant TI as Technical Indicator
    participant SA as Sentiment Analyst
    participant RM as Risk Manager
    participant TE as Trade Executor
    participant K as Kite API
    
    U->>C: Initiate Analysis
    activate C
    
    par Parallel Analysis
        C->>MA: Analyze Market Trends
        activate MA
        MA-->>C: Trend Analysis Result
        deactivate MA
    and
        C->>TI: Calculate Indicators
        activate TI
        TI-->>C: Technical Signals
        deactivate TI
    and
        C->>SA: Analyze Sentiment
        activate SA
        SA-->>C: Sentiment Score
        deactivate SA
    end
    
    C->>C: Fusion Algorithm<br/>Combine Signals
    C->>RM: Validate Trade
    activate RM
    RM->>RM: Check Risk Limits
    RM->>RM: Calculate Position Size
    RM-->>C: Risk Approval + Size
    deactivate RM
    
    alt Trade Approved
        C->>TE: Execute Trade
        activate TE
        TE->>K: Place Order
        activate K
        K-->>TE: Order Confirmation
        deactivate K
        TE-->>C: Execution Status
        deactivate TE
        C-->>U: Trade Executed
    else Trade Rejected
        C-->>U: Trade Rejected (Risk)
    end
    
    deactivate C
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Ingress"
            IG[Ingress Controller<br/>SSL Termination]
        end
        
        subgraph "Application Pods"
            API1[API Pod 1]
            API2[API Pod 2]
            API3[API Pod 3]
            WS1[WebSocket Pod 1]
            WS2[WebSocket Pod 2]
        end
        
        subgraph "Agent Pods"
            AG1[Agent Pod 1<br/>MA + TI]
            AG2[Agent Pod 2<br/>SA + RM]
            AG3[Agent Pod 3<br/>TE + DP]
            CO[Coordinator Pod]
        end
        
        subgraph "Worker Pods"
            WK1[Worker Pod 1]
            WK2[Worker Pod 2]
        end
        
        subgraph "StatefulSets"
            PG[(PostgreSQL<br/>Primary)]
            PGR[(PostgreSQL<br/>Replica)]
            RD[(Redis<br/>Master)]
            RDR[(Redis<br/>Replica)]
        end
        
        subgraph "Monitoring"
            PR[Prometheus]
            GR[Grafana]
            LK[Loki]
        end
    end
    
    subgraph "External"
        LB[Load Balancer]
        CDN[CDN]
        S3[Object Storage]
    end
    
    CDN --> LB
    LB --> IG
    IG --> API1
    IG --> API2
    IG --> API3
    IG --> WS1
    IG --> WS2
    
    API1 --> CO
    API2 --> CO
    API3 --> CO
    
    CO --> AG1
    CO --> AG2
    CO --> AG3
    
    AG1 --> PG
    AG2 --> PG
    AG3 --> PG
    
    API1 --> RD
    API2 --> RD
    API3 --> RD
    
    PG --> PGR
    RD --> RDR
    
    API1 --> S3
    WK1 --> S3
    
    PR --> API1
    PR --> AG1
    PR --> PG
    GR --> PR
    LK --> API1
    
    style IG fill:#ffeb3b
    style PG fill:#e8f5e9
    style RD fill:#e8f5e9
    style PR fill:#ffccbc
    style GR fill:#ffccbc
```

## Database Schema

```mermaid
erDiagram
    USERS ||--o{ PORTFOLIOS : has
    USERS ||--o{ API_KEYS : has
    USERS ||--o{ SESSIONS : has
    
    PORTFOLIOS ||--o{ POSITIONS : contains
    PORTFOLIOS ||--o{ TRADES : executes
    
    TRADES ||--|| ORDERS : creates
    TRADES }o--|| SYMBOLS : involves
    
    POSITIONS }o--|| SYMBOLS : holds
    
    SYMBOLS ||--o{ MARKET_DATA : generates
    SYMBOLS ||--o{ TECHNICAL_DATA : has
    
    AGENT_DECISIONS ||--|| TRADES : influences
    AGENT_DECISIONS }o--|| AGENTS : made_by
    
    RISK_METRICS ||--|| PORTFOLIOS : monitors
    
    USERS {
        uuid id PK
        string email UK
        string password_hash
        timestamp created_at
        timestamp updated_at
        boolean is_active
    }
    
    PORTFOLIOS {
        uuid id PK
        uuid user_id FK
        decimal total_value
        decimal cash_balance
        decimal day_pnl
        timestamp updated_at
    }
    
    POSITIONS {
        uuid id PK
        uuid portfolio_id FK
        string symbol FK
        integer quantity
        decimal average_price
        decimal current_price
        decimal unrealized_pnl
        timestamp opened_at
    }
    
    TRADES {
        uuid id PK
        uuid portfolio_id FK
        string symbol FK
        string trade_type
        integer quantity
        decimal entry_price
        decimal exit_price
        decimal pnl
        timestamp entry_time
        timestamp exit_time
    }
    
    ORDERS {
        uuid id PK
        uuid trade_id FK
        string order_id
        string status
        decimal price
        integer quantity
        timestamp placed_at
        timestamp filled_at
    }
    
    SYMBOLS {
        string symbol PK
        string exchange
        string name
        string sector
        boolean is_active
    }
    
    MARKET_DATA {
        uuid id PK
        string symbol FK
        decimal open
        decimal high
        decimal low
        decimal close
        bigint volume
        timestamp timestamp
    }
    
    AGENT_DECISIONS {
        uuid id PK
        uuid agent_id FK
        uuid trade_id FK
        string decision_type
        jsonb analysis_data
        decimal confidence
        timestamp created_at
    }
```

## API Architecture

```mermaid
graph TD
    subgraph "API Gateway Layer"
        GW[API Gateway<br/>Kong/AWS API GW]
        RL[Rate Limiting]
        AU[Authentication]
        CA[Caching]
    end
    
    subgraph "Service Layer"
        AS[Auth Service<br/>JWT/OAuth2]
        TS[Trading Service<br/>Order Management]
        MS[Market Service<br/>Data & Quotes]
        PS[Portfolio Service<br/>Positions & P&L]
        AGS[Agent Service<br/>AI Analysis]
    end
    
    subgraph "Business Logic"
        TL[Trading Logic]
        RL2[Risk Logic]
        AL[Agent Logic]
        DL[Data Logic]
    end
    
    subgraph "Data Access Layer"
        ORM[SQLAlchemy ORM]
        RDS[Redis Client]
        MQ[Message Queue]
    end
    
    subgraph "External APIs"
        KT[Kite Trading API]
        OA[OpenAI API]
        NA[News APIs]
    end
    
    GW --> RL
    RL --> AU
    AU --> CA
    
    CA --> AS
    CA --> TS
    CA --> MS
    CA --> PS
    CA --> AGS
    
    AS --> TL
    TS --> TL
    MS --> DL
    PS --> RL2
    AGS --> AL
    
    TL --> ORM
    RL2 --> ORM
    AL --> ORM
    DL --> RDS
    
    TL --> KT
    AL --> OA
    DL --> NA
    
    ORM --> MQ
    RDS --> MQ
    
    style GW fill:#ffeb3b
    style AS fill:#e1f5fe
    style TS fill:#e1f5fe
    style MS fill:#e1f5fe
    style PS fill:#e1f5fe
    style AGS fill:#e1f5fe
```

## Security Architecture

```mermaid
graph TB
    subgraph "External Layer"
        CF[CloudFlare<br/>DDoS Protection]
        WAF[Web Application<br/>Firewall]
    end
    
    subgraph "Network Layer"
        VPC[Virtual Private Cloud]
        SG[Security Groups]
        NACL[Network ACLs]
    end
    
    subgraph "Application Security"
        JWT[JWT Tokens<br/>15min expiry]
        RBAC[Role-Based<br/>Access Control]
        ENC[Encryption<br/>TLS 1.3]
        CSRF[CSRF Protection]
    end
    
    subgraph "Data Security"
        DBE[DB Encryption<br/>at Rest]
        FLE[Field-Level<br/>Encryption]
        BKP[Encrypted<br/>Backups]
    end
    
    subgraph "Secret Management"
        HSM[Hardware Security<br/>Module]
        KMS[Key Management<br/>Service]
        VAULT[HashiCorp Vault]
    end
    
    subgraph "Monitoring & Audit"
        SIEM[SIEM System]
        IDS[Intrusion Detection]
        AUDIT[Audit Logs]
    end
    
    CF --> WAF
    WAF --> VPC
    VPC --> SG
    SG --> NACL
    
    NACL --> JWT
    JWT --> RBAC
    RBAC --> ENC
    ENC --> CSRF
    
    CSRF --> DBE
    DBE --> FLE
    FLE --> BKP
    
    HSM --> KMS
    KMS --> VAULT
    VAULT --> JWT
    VAULT --> DBE
    
    JWT --> AUDIT
    RBAC --> AUDIT
    DBE --> AUDIT
    AUDIT --> SIEM
    IDS --> SIEM
    
    style CF fill:#ffcdd2
    style WAF fill:#ffcdd2
    style HSM fill:#c8e6c9
    style VAULT fill:#c8e6c9
    style SIEM fill:#fff9c4
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Frontend"
        UI[React UI]
        WC[WebSocket Client]
        RC[REST Client]
    end
    
    subgraph "Backend Services"
        API[FastAPI]
        WSS[WebSocket Server]
        SCH[Scheduler]
    end
    
    subgraph "Agent System"
        CREW[CrewAI Manager]
        AGENTS[AI Agents]
    end
    
    subgraph "Infrastructure"
        CACHE[(Redis)]
        DB[(PostgreSQL)]
        QUEUE[Celery/RabbitMQ]
    end
    
    subgraph "External"
        KITE[Kite API]
        AI[AI Services]
    end
    
    UI --> RC
    UI --> WC
    RC --> API
    WC --> WSS
    
    API --> CREW
    WSS --> CACHE
    SCH --> CREW
    
    CREW --> AGENTS
    AGENTS --> AI
    AGENTS --> CACHE
    AGENTS --> DB
    
    API --> DB
    API --> CACHE
    API --> QUEUE
    
    AGENTS --> KITE
    API --> KITE
    
    QUEUE --> SCH
    
    style UI fill:#e1f5fe
    style API fill:#fff3e0
    style CREW fill:#f3e5f5
    style KITE fill:#fff9c4
```

## Monitoring Architecture

```mermaid
graph TD
    subgraph "Application Metrics"
        APP[Application<br/>Instrumentation]
        API_M[API Metrics]
        DB_M[Database Metrics]
        CACHE_M[Cache Metrics]
    end
    
    subgraph "Collection Layer"
        PROM[Prometheus<br/>Time Series DB]
        LOKI[Loki<br/>Log Aggregation]
        TEMPO[Tempo<br/>Distributed Tracing]
    end
    
    subgraph "Visualization"
        GRAF[Grafana<br/>Dashboards]
        ALERT[Alert Manager]
    end
    
    subgraph "Notifications"
        EMAIL[Email]
        SLACK[Slack]
        PD[PagerDuty]
    end
    
    APP --> PROM
    API_M --> PROM
    DB_M --> PROM
    CACHE_M --> PROM
    
    APP --> LOKI
    APP --> TEMPO
    
    PROM --> GRAF
    LOKI --> GRAF
    TEMPO --> GRAF
    
    GRAF --> ALERT
    ALERT --> EMAIL
    ALERT --> SLACK
    ALERT --> PD
    
    style PROM fill:#ff6f00
    style GRAF fill:#00e676
    style ALERT fill:#ff5252
```

---

## How to Use These Diagrams

1. **Viewing**: These diagrams can be viewed in:
   - GitHub (automatic rendering)
   - VS Code with Mermaid extension
   - Online Mermaid editors
   - Any Markdown viewer with Mermaid support

2. **Exporting**: To export as images:
   - Use Mermaid CLI: `mmdc -i diagram.mmd -o diagram.png`
   - Use online tools like mermaid.live
   - Screenshot from rendered view

3. **Customizing**: 
   - Colors can be changed in the `style` declarations
   - Add more nodes by following the existing patterns
   - Modify relationships by changing arrow types

4. **Integration**: 
   - Include in documentation
   - Use in presentations
   - Add to README files
   - Embed in wiki pages

These diagrams provide a comprehensive visual representation of the Shagun Intelligence architecture and can be updated as the system evolves.