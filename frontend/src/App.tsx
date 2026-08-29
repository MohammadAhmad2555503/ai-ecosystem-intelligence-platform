import { useEffect, useState } from "react";
import "./App.css";

type ServiceStatus = "checking" | "online" | "offline";

type SourceItem = {
  title: string;
  subtitle: string;
  score: number;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources: SourceItem[];
  confidence: string;
  route: string;
};

const MODEL_API_URL = "/api/model";
const ROUTER_API_URL = "/api/router";

const defaultSources: SourceItem[] = [
  {
    title: "AI Ecosystem Knowledge Graph",
    subtitle: "Dataset 3",
    score: 0.98
  },
  {
    title: "Research Literature Corpus",
    subtitle: "Dataset 1",
    score: 0.96
  },
  {
    title: "GitHub and Hugging Face Artefacts",
    subtitle: "Dataset 2",
    score: 0.95
  }
];

const risingTopics = [
  { name: "Long Context Models", growth: "230%" },
  { name: "Multimodal external systems", growth: "178%" },
  { name: "AI Agents", growth: "156%" },
  { name: "RAG Systems", growth: "134%" },
  { name: "Model Alignment", growth: "112%" }
];

const recentPapers = [
  "A Survey on Hallucination in external systems",
  "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
  "Constitutional AI: Harmlessness from AI Feedback"
];

function App() {
  const [modelStatus, setModelStatus] = useState<ServiceStatus>("checking");
  const [routerStatus, setRouterStatus] = useState<ServiceStatus>("checking");
  const [question, setQuestion] = useState("");
  const [useRouter, setUseRouter] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      text: "Ask a question about AI papers, RAG trends, influential organisations, GitHub repositories, Hugging Face models, or ecosystem signals.",
      sources: defaultSources,
      confidence: "Ready",
      route: "Research Assistant"
    }
  ]);

  useEffect(() => {
    checkBackendHealth();
  }, []);

  async function checkBackendHealth() {
    const modelHealth = await checkHealth(`${MODEL_API_URL}/health`);
    const routerHealth = await checkHealth(`${ROUTER_API_URL}/health`);

    setModelStatus(modelHealth);
    setRouterStatus(routerHealth);
  }

  async function checkHealth(url: string): Promise<ServiceStatus> {
    try {
      const response = await fetch(url);
      return response.ok ? "online" : "offline";
    } catch {
      return "offline";
    }
  }

  async function askQuestion() {
    const cleanQuestion = question.trim();

    if (!cleanQuestion || isLoading) {
      return;
    }

    setQuestion("");
    setIsLoading(true);

    setMessages((currentMessages) => [
      ...currentMessages,
      buildUserMessage(cleanQuestion)
    ]);

    const assistantMessage = await getRagAnswer(cleanQuestion);

    setMessages((currentMessages) => [
      ...currentMessages,
      assistantMessage
    ]);

    setIsLoading(false);
  }

  async function getRagAnswer(cleanQuestion: string): Promise<ChatMessage> {
    const endpoint = useRouter
      ? `${ROUTER_API_URL}/rag-answer`
      : `${MODEL_API_URL}/rag-answer`;

    const requestBody = useRouter
      ? { question: cleanQuestion, top_k: 3, user_id: "frontend-demo-user" }
      : { question: cleanQuestion, top_k: 3 };

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody)
      });

      const payload = await response.json();
      return buildAssistantMessage(payload, useRouter);
    } catch {
      return buildFallbackMessage(cleanQuestion, useRouter);
    }
  }

  function buildUserMessage(text: string): ChatMessage {
    return {
      id: crypto.randomUUID(),
      role: "user",
      text,
      sources: [],
      confidence: "",
      route: ""
    };
  }

  function buildAssistantMessage(payload: Record<string, unknown>, routed: boolean): ChatMessage {
    return {
      id: crypto.randomUUID(),
      role: "assistant",
      text: getAnswerText(payload),
      sources: getSources(payload),
      confidence: getStringValue(payload, ["confidence", "confidence_label"], "High Confidence"),
      route: getStringValue(payload, ["route", "variant", "model_variant"], routed ? "A/B Router" : "Model API")
    };
  }

  function buildFallbackMessage(cleanQuestion: string, routed: boolean): ChatMessage {
    return {
      id: crypto.randomUUID(),
      role: "assistant",
      text: `The backend is currently unreachable, so the frontend is running in demo mode. Your question was: "${cleanQuestion}". When Docker is running, this panel will show live grounded RAG answers from your FastAPI backend.`,
      sources: defaultSources,
      confidence: "Demo Mode",
      route: routed ? "Router Offline" : "Model API Offline"
    };
  }

  function getAnswerText(payload: Record<string, unknown>) {
    return getStringValue(
      payload,
      ["answer", "response", "result", "message"],
      "The backend responded, but no answer field was found."
    );
  }

  function getStringValue(payload: Record<string, unknown>, keys: string[], fallback: string) {
    for (const key of keys) {
      const value = payload[key];

      if (typeof value === "string" && value.trim().length > 0) {
        return value;
      }
    }

    return fallback;
  }

  function getSources(payload: Record<string, unknown>): SourceItem[] {
    const rawSources = payload.sources || payload.context || payload.documents || payload.results;

    if (!Array.isArray(rawSources)) {
      return defaultSources;
    }

    return rawSources.slice(0, 3).map((source, index) => normaliseSource(source, index));
  }

  function normaliseSource(source: unknown, index: number): SourceItem {
    if (!isObject(source)) {
      return {
        title: `Retrieved Source ${index + 1}`,
        subtitle: "AI ecosystem record",
        score: 0.94
      };
    }

    return {
      title: getStringValue(source, ["title", "paper_title", "name"], `Retrieved Source ${index + 1}`),
      subtitle: getStringValue(source, ["year", "source", "platform"], "AI ecosystem record"),
      score: getNumericValue(source, ["score", "similarity", "relevance"], 0.94)
    };
  }

  function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
  }

  function getNumericValue(payload: Record<string, unknown>, keys: string[], fallback: number) {
    for (const key of keys) {
      const value = Number(payload[key]);

      if (!Number.isNaN(value)) {
        return value;
      }
    }

    return fallback;
  }

  return (
    <main className="app-shell">
      <Sidebar />

      <section className="main-area">
        <TopBar />

        <div className="content-grid">
          <section className="left-column">
            <HeroPanel />
            <FeatureCards />

            <section className="chat-panel">
              <div className="chat-header">
                <div>
                  <h2>AI Research Assistant</h2>
                  <p>Source-grounded answers from your AI ecosystem backend.</p>
                </div>

                <label className="router-toggle">
                  <input
                    type="checkbox"
                    checked={useRouter}
                    onChange={(event) => setUseRouter(event.target.checked)}
                  />
                  Use A/B Router
                </label>
              </div>

              <div className="messages">
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}

                {isLoading && (
                  <div className="assistant-card">
                    Thinking through the research corpus...
                  </div>
                )}
              </div>

              <div className="question-box">
                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      askQuestion();
                    }
                  }}
                  placeholder="Ask a research question..."
                />

                <button onClick={askQuestion}>➤</button>
              </div>
            </section>
          </section>

          <aside className="right-column">
            <Panel title="System Status">
              <StatusRow label="Model API" status={modelStatus} />
              <StatusRow label="A/B Router" status={routerStatus} />
              <StatusRow label="Frontend" status="online" />
            </Panel>

            <Panel title="Knowledge Base Overview">
              <div className="knowledge-layout">
                <div className="knowledge-ring">
                  <strong>4,124</strong>
                  <span>Papers</span>
                </div>

                <div>
                  <InfoRow label="AI & ML" value="1,842" />
                  <InfoRow label="external systems" value="1,256" />
                  <InfoRow label="NLP" value="642" />
                  <InfoRow label="Multimodal" value="384" />
                </div>
              </div>
            </Panel>

            <Panel title="Research Trend Radar">
              <TrendRadar />
            </Panel>

            <Panel title="Top Rising Topics">
              {risingTopics.map((topic) => (
                <TopicRow key={topic.name} name={topic.name} growth={topic.growth} />
              ))}
            </Panel>

            <Panel title="Recent Additions">
              {recentPapers.map((paper) => (
                <article className="paper-card" key={paper}>
                  {paper}
                  <span>AI Research · 2024</span>
                </article>
              ))}
            </Panel>
          </aside>
        </div>
      </section>
    </main>
  );
}

function Sidebar() {
  const navigationItems = [
    "Home",
    "Explore Papers",
    "Topical Maps",
    "Trend Radar",
    "Authors",
    "Compare Papers",
    "Saved Insights"
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">✦</div>
        <div>
          <h1>AI Research RAG</h1>
          <p>Explore. Ask. Discover.</p>
        </div>
      </div>

      <button className="new-chat-button">＋ New Research Chat</button>

      <nav>
        {navigationItems.map((item, index) => (
          <div className={index === 0 ? "nav-item active" : "nav-item"} key={item}>
            <span>◈</span>
            {item}
          </div>
        ))}
      </nav>

      <div className="library-card">
        <p>MY LIBRARY</p>
        <InfoRow label="All Papers" value="4,124" />
        <InfoRow label="Recently Added" value="142" />
        <InfoRow label="Favourites" value="89" />
        <InfoRow label="Collections" value="12" />
      </div>

      <div className="graph-card">
        <strong>✦ Knowledge Graph</strong>
        <p>Visualise connections between papers, organisations, topics, and models.</p>
        <button>Open Graph →</button>
      </div>
    </aside>
  );
}

function TopBar() {
  return (
    <header className="topbar">
      <div className="search-bar">
        ⌕ Search for papers, concepts, authors, topics...
      </div>

      <div className="top-actions">
        <button>✨ Deep Research</button>
        <button>☾</button>
        <button>♢</button>
        <div className="avatar" />
      </div>
    </header>
  );
}

function HeroPanel() {
  return (
    <section className="hero-panel">
      <div>
        <h2>
          <span>Good Evening,</span> Ahmad 👋
        </h2>
        <p>
          Your AI research ecosystem. <strong>4,124 papers</strong> and growing.
        </p>
      </div>

      <div className="brain-icon">🧠</div>
    </section>
  );
}

function FeatureCards() {
  const features = [
    ["Smart RAG", "Ask questions and get grounded answers."],
    ["Paper Explorer", "Discover papers by topics and themes."],
    ["Trend Radar", "Track emerging AI research signals."],
    ["Topic Maps", "Visualise relationships between concepts."]
  ];

  return (
    <section className="feature-grid">
      {features.map(([title, text]) => (
        <article className="feature-card" key={title}>
          <strong>{title}</strong>
          <p>{text}</p>
        </article>
      ))}
    </section>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return <div className="user-message">{message.text}</div>;
  }

  return (
    <article className="assistant-card">
      <div className="answer-header">
        <strong>✨ AI Research Assistant</strong>
        {message.confidence && <span>{message.confidence}</span>}
        {message.route && <span>{message.route}</span>}
      </div>

      <p>{message.text}</p>

      {message.sources.length > 0 && (
        <div className="source-section">
          <small>Sources</small>

          <div className="source-grid">
            {message.sources.map((source) => (
              <div className="source-card" key={source.title}>
                <strong>{source.title}</strong>
                <span>{source.subtitle}</span>
                <em>{Math.round(source.score * 100)}% relevance</em>
              </div>
            ))}
          </div>
        </div>
      )}
    </article>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="side-panel">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusRow({ label, status }: { label: string; status: ServiceStatus }) {
  return (
    <div className="info-row">
      <span>{label}</span>
      <strong className={status}>{status}</strong>
    </div>
  );
}

function TopicRow({ name, growth }: { name: string; growth: string }) {
  return (
    <div className="topic-row">
      <span>{name}</span>
      <i />
      <strong>↑ {growth}</strong>
    </div>
  );
}

function TrendRadar() {
  return (
    <div className="radar">
      <div />
      <div />
      <div />
      <span className="radar-top">external systems</span>
      <span className="radar-left">RAG</span>
      <span className="radar-bottom">Agents</span>
      <span className="radar-right">Multimodal</span>
    </div>
  );
}

export default App;



