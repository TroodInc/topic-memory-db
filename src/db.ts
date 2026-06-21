import pg from "pg";
import { randomUUID } from "node:crypto";
import type {
  StoredPost,
  StoredArticle,
  ArticleExtractionJob,
  StoredEmbedding,
  StoredTopic,
  TopicArticleLink,
  ContentPlanItem,
  PublishedArticle,
  DraftArticle,
  StoredSource,
  StoredInterest,
  ArticleInterest,
  ArticleFeedback,
} from "./types.js";

/** Embedding dimension for text-embedding-3-small */
const EMBEDDING_DIM = 1536;

function assertFiniteVector(vector: number[], label: string): number[] {
  if (vector.length !== EMBEDDING_DIM) {
    throw new Error(`${label} must have length ${EMBEDDING_DIM}, got ${vector.length}`);
  }

  for (let i = 0; i < vector.length; i++) {
    const value = vector[i];
    if (!Number.isFinite(value)) {
      throw new Error(`${label} contains non-finite value at index ${i}`);
    }
  }

  return vector;
}

function parseVector(value: unknown, label: string): number[] {
  // pg returns float8[] as a JS array of numbers directly
  if (Array.isArray(value)) {
    return assertFiniteVector(value.map((item) => Number(item)), label);
  }
  // Fallback: Postgres array literal format {0.1,0.2,...}
  if (typeof value === "string") {
    const trimmed = value.trim();
    const inner = trimmed.startsWith("{") && trimmed.endsWith("}")
      ? trimmed.slice(1, -1)
      : trimmed.startsWith("[") && trimmed.endsWith("]")
        ? trimmed.slice(1, -1)
        : null;
    if (!inner) throw new Error(`${label} has invalid vector format`);
    return assertFiniteVector(inner.split(",").map((x) => Number(x.trim())), label);
  }
  throw new Error(`${label} has unsupported vector type`);
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return normA === 0 || normB === 0 ? 0 : dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function normalizeTimestamp(value: unknown): number | null {
  if (value === undefined || value === null || value === "") return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return null;
    if (/^\d+$/.test(trimmed)) {
      const parsedNumber = Number(trimmed);
      return Number.isFinite(parsedNumber) ? parsedNumber : null;
    }
    const parsedDate = Date.parse(trimmed);
    return Number.isNaN(parsedDate) ? null : parsedDate;
  }
  return null;
}

const EXTENSIONS = `
  CREATE SCHEMA IF NOT EXISTS content_engine;
  CREATE EXTENSION IF NOT EXISTS pgcrypto;
`;

const SCHEMA = `

  CREATE TABLE IF NOT EXISTS posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    telegram_id BIGINT NOT NULL,
    channel_id TEXT NOT NULL,
    text TEXT NOT NULL,
    urls JSONB NOT NULL DEFAULT '[]',
    date BIGINT NOT NULL,
    processed_at BIGINT NOT NULL,
    UNIQUE(telegram_id, channel_id)
  );

  CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    adapter_type TEXT NOT NULL,
    adapter_config JSONB NOT NULL DEFAULT '{}',
    interest_ids JSONB NOT NULL DEFAULT '[]',
    last_ingested_at BIGINT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS interests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding float8[] NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID REFERENCES posts(id),
    source_id UUID REFERENCES sources(id),
    external_id TEXT,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    word_count INTEGER NOT NULL,
    published_at BIGINT,
    processed_at BIGINT NOT NULL
  );

  CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_source_external
    ON articles (source_id, external_id)
    WHERE source_id IS NOT NULL AND external_id IS NOT NULL;

  CREATE TABLE IF NOT EXISTS article_interests (
    article_id UUID NOT NULL REFERENCES articles(id),
    interest_id UUID NOT NULL REFERENCES interests(id),
    score DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (article_id, interest_id)
  );

  CREATE INDEX IF NOT EXISTS idx_article_interests_interest
    ON article_interests (interest_id, score DESC);

  CREATE TABLE IF NOT EXISTS article_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    article_id UUID NOT NULL REFERENCES articles(id),
    interest_id UUID REFERENCES interests(id),
    signal TEXT NOT NULL CHECK (signal IN ('like', 'less', 'skip')),
    created_at BIGINT NOT NULL
  );

  CREATE INDEX IF NOT EXISTS idx_article_feedback_user
    ON article_feedback (user_id, created_at DESC);

  CREATE TABLE IF NOT EXISTS article_extraction_queue (
    url TEXT PRIMARY KEY,
    post_id UUID NOT NULL REFERENCES posts(id),
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    next_attempt_at BIGINT NOT NULL,
    last_attempt_at BIGINT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
  );

  CREATE INDEX IF NOT EXISTS idx_article_extraction_queue_next_attempt
    ON article_extraction_queue (status, next_attempt_at);

  CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL UNIQUE REFERENCES articles(id),
    embedding float8[] NOT NULL,
    model TEXT NOT NULL,
    created_at BIGINT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    centroid_embedding float8[] NOT NULL,
    article_count INTEGER NOT NULL DEFAULT 0,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS topic_articles (
    topic_id UUID NOT NULL REFERENCES topics(id),
    article_id UUID NOT NULL REFERENCES articles(id),
    similarity DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (topic_id, article_id)
  );

  CREATE TABLE IF NOT EXISTS content_plan (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id UUID NOT NULL REFERENCES topics(id),
    status TEXT NOT NULL DEFAULT 'draft',
    priority INTEGER NOT NULL DEFAULT 0,
    human_comment TEXT,
    scheduled_date BIGINT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS draft_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_plan_id UUID NOT NULL UNIQUE REFERENCES content_plan(id),
    topic_id UUID NOT NULL REFERENCES topics(id),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]',
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS published_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_plan_id UUID NOT NULL REFERENCES content_plan(id),
    discourse_topic_id INTEGER NOT NULL,
    discourse_post_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    published_at BIGINT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS pipeline_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at BIGINT NOT NULL
  );
`;

/**
 * PostgreSQL + pgvector semantic topic memory database.
 *
 * Stores posts, articles, embeddings (as native vectors), topics,
 * content plans, and publication state. Supports vector similarity
 * search via pgvector for topic clustering and nearest-neighbor queries.
 *
 * All methods are async.
 */
export class TopicMemoryDB {
  private pool: pg.Pool;
  private initialized = false;

  constructor(connectionString: string) {
    this.pool = new pg.Pool({
      connectionString,
      options: '-csearch_path=content_engine',
    });
  }

  /** Initialize schema and register pgvector type. Call once before use. */
  async init(): Promise<void> {
    if (this.initialized) return;
    const client = await this.pool.connect();
    try {
      await client.query(EXTENSIONS);
      await client.query(SCHEMA);
    } finally {
      client.release();
    }
    this.initialized = true;
  }

  // --- Posts ---

  /** Insert a post. Returns null if already exists (duplicate). */
  async insertPost(post: Omit<StoredPost, "id" | "processedAt">): Promise<StoredPost | null> {
    const id = randomUUID();
    const now = Date.now();
    try {
      await this.pool.query(
        `INSERT INTO posts (id, telegram_id, channel_id, text, urls, date, processed_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7)`,
        [id, post.telegramId, post.channelId, post.text, JSON.stringify(post.urls), post.date, now]
      );
      return { id, ...post, processedAt: now };
    } catch (e: unknown) {
      if (e instanceof Error && e.message.includes("duplicate key")) return null;
      throw e;
    }
  }

  /** Check if a post with the given Telegram ID already exists */
  async hasPost(telegramId: number, channelId: string): Promise<boolean> {
    const { rows } = await this.pool.query(
      "SELECT 1 FROM posts WHERE telegram_id = $1 AND channel_id = $2",
      [telegramId, channelId]
    );
    return rows.length > 0;
  }

  /** Get a post by its Telegram ID and channel */
  async getPostByTelegramId(telegramId: number, channelId: string): Promise<StoredPost | null> {
    const { rows } = await this.pool.query(
      "SELECT * FROM posts WHERE telegram_id = $1 AND channel_id = $2",
      [telegramId, channelId]
    );
    if (rows.length === 0) return null;
    return this.mapPost(rows[0]);
  }

  /** Get all Telegram post IDs for a channel */
  async getPostIds(channelId: string): Promise<number[]> {
    const { rows } = await this.pool.query(
      "SELECT telegram_id FROM posts WHERE channel_id = $1 ORDER BY telegram_id",
      [channelId]
    );
    return rows.map((r: { telegram_id: string }) => Number(r.telegram_id));
  }

  // --- Articles ---

  /** Insert an article. Returns null if URL or (source_id, external_id) already exists. */
  async insertArticle(article: Omit<StoredArticle, "id" | "processedAt">): Promise<StoredArticle | null> {
    const id = randomUUID();
    const now = Date.now();
    try {
      await this.pool.query(
        `INSERT INTO articles
           (id, post_id, source_id, external_id, url, title, content, summary, word_count, published_at, processed_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
        [
          id,
          article.postId ?? null,
          article.sourceId ?? null,
          article.externalId ?? null,
          article.url,
          article.title,
          article.content,
          article.summary ?? null,
          article.wordCount,
          article.publishedAt ?? null,
          now,
        ]
      );
      return { id, ...article, processedAt: now };
    } catch (e: unknown) {
      if (e instanceof Error && e.message.includes("duplicate key")) return null;
      throw e;
    }
  }

  /** Check if an article with (source_id, external_id) already exists */
  async hasArticleByExternalId(sourceId: string, externalId: string): Promise<boolean> {
    const { rows } = await this.pool.query(
      "SELECT 1 FROM articles WHERE source_id = $1 AND external_id = $2",
      [sourceId, externalId]
    );
    return rows.length > 0;
  }

  /** Check if an article with the given URL already exists */
  async hasArticle(url: string): Promise<boolean> {
    const { rows } = await this.pool.query("SELECT 1 FROM articles WHERE url = $1", [url]);
    return rows.length > 0;
  }

  /** Get an article by URL */
  async getArticleByUrl(url: string): Promise<StoredArticle | null> {
    const { rows } = await this.pool.query("SELECT * FROM articles WHERE url = $1", [url]);
    if (rows.length === 0) return null;
    return this.mapArticle(rows[0]);
  }

  /** Get an article by its database ID */
  async getArticleById(id: string): Promise<StoredArticle | null> {
    const { rows } = await this.pool.query("SELECT * FROM articles WHERE id = $1", [id]);
    if (rows.length === 0) return null;
    return this.mapArticle(rows[0]);
  }

  /** Get articles without embeddings */
  async getArticlesWithoutEmbeddings(): Promise<StoredArticle[]> {
    const { rows } = await this.pool.query(
      `SELECT a.* FROM articles a
       LEFT JOIN embeddings e ON e.article_id = a.id
       WHERE e.id IS NULL`
    );
    return rows.map((r: Record<string, unknown>) => this.mapArticle(r));
  }

  async getAllArticles(): Promise<StoredArticle[]> {
    const { rows } = await this.pool.query(
      "SELECT * FROM articles ORDER BY processed_at DESC"
    );
    return rows.map((r: Record<string, unknown>) => this.mapArticle(r));
  }

  async enqueueArticleExtractionJob(job: Pick<ArticleExtractionJob, "url" | "postId">): Promise<void> {
    const now = Date.now();
    await this.pool.query(
      `INSERT INTO article_extraction_queue
         (url, post_id, status, attempt_count, last_error, next_attempt_at, last_attempt_at, created_at, updated_at)
       VALUES ($1, $2, 'pending', 0, NULL, $3, NULL, $3, $3)
       ON CONFLICT (url) DO UPDATE SET
         post_id = EXCLUDED.post_id,
         status = CASE WHEN article_extraction_queue.status = 'failed' THEN 'pending' ELSE article_extraction_queue.status END,
         next_attempt_at = LEAST(article_extraction_queue.next_attempt_at, EXCLUDED.next_attempt_at),
         updated_at = EXCLUDED.updated_at`,
      [job.url, job.postId, now]
    );
  }

  async getPendingArticleExtractionJobs(limit = 100): Promise<ArticleExtractionJob[]> {
    const { rows } = await this.pool.query(
      `SELECT * FROM article_extraction_queue
       WHERE status IN ('pending', 'retry') AND next_attempt_at <= $1
       ORDER BY next_attempt_at ASC, created_at ASC
       LIMIT $2`,
      [Date.now(), limit]
    );
    return rows.map((row: Record<string, unknown>) => this.mapArticleExtractionJob(row));
  }

  async recordArticleExtractionFailure(url: string, lastError: string): Promise<void> {
    const { rows } = await this.pool.query(
      "SELECT attempt_count FROM article_extraction_queue WHERE url = $1",
      [url]
    );
    if (rows.length === 0) return;

    const now = Date.now();
    const attemptCount = Number(rows[0].attempt_count) + 1;
    const status: ArticleExtractionJob["status"] = attemptCount >= 5 ? "failed" : "retry";
    const nextAttemptAt = now + this.articleExtractionBackoffMs(attemptCount);
    await this.pool.query(
      `UPDATE article_extraction_queue
       SET status = $2,
           attempt_count = $3,
           last_error = $4,
           next_attempt_at = $5,
           last_attempt_at = $6,
           updated_at = $6
       WHERE url = $1`,
      [url, status, attemptCount, lastError, nextAttemptAt, now]
    );
  }

  async completeArticleExtractionJob(url: string): Promise<void> {
    await this.pool.query("DELETE FROM article_extraction_queue WHERE url = $1", [url]);
  }

  /** Update article summary */
  async updateArticleSummary(articleId: string, summary: string): Promise<void> {
    await this.pool.query("UPDATE articles SET summary = $1 WHERE id = $2", [summary, articleId]);
  }

  // --- Embeddings ---

  /** Insert an embedding (upsert by article_id) */
  async insertEmbedding(embedding: Omit<StoredEmbedding, "id" | "createdAt">): Promise<StoredEmbedding> {
    const id = randomUUID();
    const now = Date.now();
    const vec = assertFiniteVector(embedding.embedding, "embedding");
    await this.pool.query(
      `INSERT INTO embeddings (id, article_id, embedding, model, created_at)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (article_id) DO UPDATE SET
         embedding = EXCLUDED.embedding,
         model = EXCLUDED.model,
         created_at = EXCLUDED.created_at`,
      [id, embedding.articleId, vec, embedding.model, now]
    );
    return { id, ...embedding, createdAt: now };
  }

  /** Get all embeddings */
  async getAllEmbeddings(): Promise<StoredEmbedding[]> {
    const { rows } = await this.pool.query("SELECT * FROM embeddings");
    return rows.map((r: Record<string, unknown>) => this.mapEmbedding(r));
  }

  /** Get embedding for an article */
  async getEmbeddingByArticleId(articleId: string): Promise<StoredEmbedding | null> {
    const { rows } = await this.pool.query(
      "SELECT * FROM embeddings WHERE article_id = $1",
      [articleId]
    );
    if (rows.length === 0) return null;
    return this.mapEmbedding(rows[0]);
  }

  /** Find K nearest embeddings using in-memory cosine similarity (no pgvector needed). */
  async findNearestEmbeddings(
    queryEmbedding: number[],
    k = 5
  ): Promise<Array<StoredEmbedding & { distance: number }>> {
    const query = assertFiniteVector(queryEmbedding, "queryEmbedding");
    const all = await this.getAllEmbeddings();
    return all
      .map((e) => ({ ...e, distance: 1 - cosineSimilarity(query, e.embedding) }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, k);
  }

  // --- Topics ---

  /** Insert or update a topic */
  async upsertTopic(topic: Omit<StoredTopic, "createdAt" | "updatedAt">): Promise<StoredTopic> {
    const now = Date.now();
    const vec = assertFiniteVector(topic.centroidEmbedding, `topic centroid for ${topic.name}`);
    await this.pool.query(
      `INSERT INTO topics (id, name, description, centroid_embedding, article_count, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (id) DO UPDATE SET
         name = EXCLUDED.name,
         description = EXCLUDED.description,
         centroid_embedding = EXCLUDED.centroid_embedding,
         article_count = EXCLUDED.article_count,
         updated_at = EXCLUDED.updated_at`,
      [topic.id, topic.name, topic.description, vec, topic.articleCount, now, now]
    );
    return { ...topic, createdAt: now, updatedAt: now };
  }

  /** Get all topics */
  async getAllTopics(): Promise<StoredTopic[]> {
    const { rows } = await this.pool.query("SELECT * FROM topics ORDER BY updated_at DESC");
    return rows.map((r: Record<string, unknown>) => this.mapTopic(r));
  }

  /** Link an article to a topic */
  async linkArticleToTopic(link: TopicArticleLink): Promise<void> {
    await this.pool.query(
      `INSERT INTO topic_articles (topic_id, article_id, similarity)
       VALUES ($1, $2, $3)
       ON CONFLICT (topic_id, article_id) DO UPDATE SET similarity = EXCLUDED.similarity`,
      [link.topicId, link.articleId, link.similarity]
    );
  }

  /** Get article IDs linked to a topic */
  async getTopicArticleIds(topicId: string): Promise<string[]> {
    const { rows } = await this.pool.query(
      "SELECT article_id FROM topic_articles WHERE topic_id = $1 ORDER BY similarity DESC",
      [topicId]
    );
    return rows.map((r: { article_id: string }) => r.article_id);
  }

  // --- Content Plan ---

  /** Insert a content plan item */
  async insertContentPlanItem(item: Omit<ContentPlanItem, "id" | "createdAt" | "updatedAt">): Promise<ContentPlanItem> {
    const id = randomUUID();
    const now = Date.now();
    const scheduledDate = normalizeTimestamp(item.scheduledDate);
    await this.pool.query(
      `INSERT INTO content_plan (id, topic_id, status, priority, human_comment, scheduled_date, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
      [id, item.topicId, item.status, item.priority, item.humanComment || null, scheduledDate, now, now]
    );
    return { id, ...item, scheduledDate: scheduledDate ?? undefined, createdAt: now, updatedAt: now };
  }

  /** Update content plan item status */
  async updateContentPlanStatus(id: string, status: ContentPlanItem["status"]): Promise<void> {
    await this.pool.query(
      "UPDATE content_plan SET status = $1, updated_at = $2 WHERE id = $3",
      [status, Date.now(), id]
    );
  }

  async updateContentPlanItem(
    id: string,
    updates: Partial<Pick<ContentPlanItem, "status" | "priority" | "humanComment" | "scheduledDate">>
  ): Promise<void> {
    const current = await this.pool.query("SELECT * FROM content_plan WHERE id = $1", [id]);
    if (current.rows.length === 0) return;

    const row = current.rows[0] as Record<string, unknown>;
    const scheduledDate =
      updates.scheduledDate !== undefined
        ? normalizeTimestamp(updates.scheduledDate)
        : row.scheduled_date
          ? Number(row.scheduled_date)
          : null;
    await this.pool.query(
      `UPDATE content_plan
       SET status = $1,
           priority = $2,
           human_comment = $3,
           scheduled_date = $4,
           updated_at = $5
       WHERE id = $6`,
      [
        updates.status ?? (row.status as ContentPlanItem["status"]),
        updates.priority ?? Number(row.priority),
        updates.humanComment ?? ((row.human_comment as string) || null),
        scheduledDate,
        Date.now(),
        id,
      ]
    );
  }

  /** Get content plan items by status */
  async getContentPlanByStatus(status: ContentPlanItem["status"]): Promise<ContentPlanItem[]> {
    const { rows } = await this.pool.query(
      "SELECT * FROM content_plan WHERE status = $1 ORDER BY priority DESC",
      [status]
    );
    return rows.map((r: Record<string, unknown>) => this.mapContentPlanItem(r));
  }

  /** Get all content plan items */
  async getAllContentPlan(): Promise<ContentPlanItem[]> {
    const { rows } = await this.pool.query(
      "SELECT * FROM content_plan ORDER BY priority DESC, created_at DESC"
    );
    return rows.map((r: Record<string, unknown>) => this.mapContentPlanItem(r));
  }

  async insertDraftArticle(
    article: Omit<DraftArticle, "id" | "createdAt" | "updatedAt">
  ): Promise<DraftArticle> {
    const id = randomUUID();
    const now = Date.now();
    await this.pool.query(
      `INSERT INTO draft_articles (id, content_plan_id, topic_id, title, body, tags, model, status, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
       ON CONFLICT (content_plan_id) DO UPDATE SET
         topic_id = EXCLUDED.topic_id,
         title = EXCLUDED.title,
         body = EXCLUDED.body,
         tags = EXCLUDED.tags,
         model = EXCLUDED.model,
         status = EXCLUDED.status,
         updated_at = EXCLUDED.updated_at`,
      [
        id,
        article.contentPlanId,
        article.topicId,
        article.title,
        article.body,
        JSON.stringify(article.tags),
        article.model,
        article.status,
        now,
        now,
      ]
    );
    return { id, ...article, createdAt: now, updatedAt: now };
  }

  async getDraftArticleByContentPlanId(contentPlanId: string): Promise<DraftArticle | null> {
    const { rows } = await this.pool.query(
      "SELECT * FROM draft_articles WHERE content_plan_id = $1",
      [contentPlanId]
    );
    if (rows.length === 0) return null;
    return this.mapDraftArticle(rows[0]);
  }

  async getDraftArticlesByStatus(status: DraftArticle["status"]): Promise<DraftArticle[]> {
    const { rows } = await this.pool.query(
      "SELECT * FROM draft_articles WHERE status = $1 ORDER BY updated_at DESC",
      [status]
    );
    return rows.map((r: Record<string, unknown>) => this.mapDraftArticle(r));
  }

  async updateDraftArticleStatus(id: string, status: DraftArticle["status"]): Promise<void> {
    await this.pool.query(
      "UPDATE draft_articles SET status = $1, updated_at = $2 WHERE id = $3",
      [status, Date.now(), id]
    );
  }

  // --- Published Articles ---

  /** Record a published article */
  async insertPublishedArticle(article: Omit<PublishedArticle, "id" | "publishedAt">): Promise<PublishedArticle> {
    const id = randomUUID();
    const now = Date.now();
    await this.pool.query(
      `INSERT INTO published_articles (id, content_plan_id, discourse_topic_id, discourse_post_id, title, published_at)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [id, article.contentPlanId, article.discourseTopicId, article.discoursePostId, article.title, now]
    );
    return { id, ...article, publishedAt: now };
  }

  async getPublishedArticles(): Promise<PublishedArticle[]> {
    const { rows } = await this.pool.query(
      "SELECT * FROM published_articles ORDER BY published_at DESC"
    );
    return rows.map((row: Record<string, unknown>) => ({
      id: row.id as string,
      contentPlanId: row.content_plan_id as string,
      discourseTopicId: Number(row.discourse_topic_id),
      discoursePostId: Number(row.discourse_post_id),
      title: row.title as string,
      publishedAt: Number(row.published_at),
    }));
  }

  // --- Pipeline State ---

  /** Get a pipeline state value */
  async getState(key: string): Promise<string | null> {
    const { rows } = await this.pool.query(
      "SELECT value FROM pipeline_state WHERE key = $1",
      [key]
    );
    return rows.length > 0 ? (rows[0].value as string) : null;
  }

  /** Set a pipeline state value */
  async setState(key: string, value: string): Promise<void> {
    await this.pool.query(
      `INSERT INTO pipeline_state (key, value, updated_at) VALUES ($1, $2, $3)
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at`,
      [key, value, Date.now()]
    );
  }

  // --- Sources ---

  async upsertSource(source: Omit<StoredSource, "id" | "createdAt" | "updatedAt"> & { id?: string }): Promise<StoredSource> {
    const id = source.id ?? randomUUID();
    const now = Date.now();
    await this.pool.query(
      `INSERT INTO sources (id, name, url, adapter_type, adapter_config, interest_ids, last_ingested_at, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
       ON CONFLICT (url) DO UPDATE SET
         name = EXCLUDED.name,
         adapter_type = EXCLUDED.adapter_type,
         adapter_config = EXCLUDED.adapter_config,
         interest_ids = EXCLUDED.interest_ids,
         updated_at = EXCLUDED.updated_at`,
      [id, source.name, source.url, source.adapterType, JSON.stringify(source.adapterConfig),
       JSON.stringify(source.interestIds), source.lastIngestedAt ?? null, now, now]
    );
    const { rows } = await this.pool.query("SELECT * FROM sources WHERE url = $1", [source.url]);
    return this.mapSource(rows[0]);
  }

  async getSourceById(id: string): Promise<StoredSource | null> {
    const { rows } = await this.pool.query("SELECT * FROM sources WHERE id = $1", [id]);
    return rows.length > 0 ? this.mapSource(rows[0]) : null;
  }

  async getAllSources(): Promise<StoredSource[]> {
    const { rows } = await this.pool.query("SELECT * FROM sources ORDER BY created_at DESC");
    return rows.map((r: Record<string, unknown>) => this.mapSource(r));
  }

  async touchSourceIngestedAt(id: string): Promise<void> {
    await this.pool.query(
      "UPDATE sources SET last_ingested_at = $1, updated_at = $1 WHERE id = $2",
      [Date.now(), id]
    );
  }

  // --- Interests ---

  async upsertInterest(interest: Omit<StoredInterest, "id" | "createdAt" | "updatedAt"> & { id?: string }): Promise<StoredInterest> {
    const id = interest.id ?? randomUUID();
    const now = Date.now();
    const vec = assertFiniteVector(interest.embedding, `interest ${interest.slug}`);
    await this.pool.query(
      `INSERT INTO interests (id, slug, name, description, embedding, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (slug) DO UPDATE SET
         name = EXCLUDED.name,
         description = EXCLUDED.description,
         embedding = EXCLUDED.embedding,
         updated_at = EXCLUDED.updated_at`,
      [id, interest.slug, interest.name, interest.description, vec, now, now]
    );
    const { rows } = await this.pool.query("SELECT * FROM interests WHERE slug = $1", [interest.slug]);
    return this.mapInterest(rows[0]);
  }

  async getAllInterests(): Promise<StoredInterest[]> {
    const { rows } = await this.pool.query("SELECT * FROM interests ORDER BY name ASC");
    return rows.map((r: Record<string, unknown>) => this.mapInterest(r));
  }

  async getInterestBySlug(slug: string): Promise<StoredInterest | null> {
    const { rows } = await this.pool.query("SELECT * FROM interests WHERE slug = $1", [slug]);
    return rows.length > 0 ? this.mapInterest(rows[0]) : null;
  }

  // --- Article–Interest matching ---

  async upsertArticleInterest(link: ArticleInterest): Promise<void> {
    await this.pool.query(
      `INSERT INTO article_interests (article_id, interest_id, score)
       VALUES ($1, $2, $3)
       ON CONFLICT (article_id, interest_id) DO UPDATE SET score = EXCLUDED.score`,
      [link.articleId, link.interestId, link.score]
    );
  }

  /** Return articles for an interest, ranked by score then recency, with user feedback applied. */
  async getFeedForInterest(
    interestId: string,
    options: { limit?: number; offset?: number; userId?: string } = {}
  ): Promise<Array<StoredArticle & { score: number }>> {
    const { limit = 20, offset = 0, userId } = options;

    // Exclude articles the user flagged "less" in the past 30 days
    const suppressClause = userId
      ? `AND a.id NOT IN (
           SELECT article_id FROM article_feedback
           WHERE user_id = $4 AND signal = 'less'
             AND created_at > ${Date.now() - 30 * 24 * 60 * 60 * 1000}
         )`
      : "";

    const params: unknown[] = [interestId, limit, offset];
    if (userId) params.push(userId);

    const { rows } = await this.pool.query(
      `SELECT a.*, ai.score
       FROM articles a
       JOIN article_interests ai ON ai.article_id = a.id
       WHERE ai.interest_id = $1
         ${suppressClause}
       ORDER BY ai.score DESC, COALESCE(a.published_at, a.processed_at) DESC
       LIMIT $2 OFFSET $3`,
      params
    );
    return rows.map((r: Record<string, unknown>) => ({
      ...this.mapArticle(r),
      score: r.score as number,
    }));
  }

  async getArticlesWithoutInterestScores(): Promise<StoredArticle[]> {
    const { rows } = await this.pool.query(
      `SELECT a.* FROM articles a
       LEFT JOIN article_interests ai ON ai.article_id = a.id
       WHERE ai.article_id IS NULL AND a.summary IS NOT NULL`
    );
    return rows.map((r: Record<string, unknown>) => this.mapArticle(r));
  }

  // --- Feedback ---

  async insertFeedback(feedback: Omit<ArticleFeedback, "id" | "createdAt">): Promise<void> {
    const id = randomUUID();
    const now = Date.now();
    await this.pool.query(
      `INSERT INTO article_feedback (id, user_id, article_id, interest_id, signal, created_at)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [id, feedback.userId, feedback.articleId, feedback.interestId ?? null, feedback.signal, now]
    );
  }

  /** Close the connection pool */
  async close(): Promise<void> {
    await this.pool.end();
  }

  private articleExtractionBackoffMs(attemptCount: number): number {
    return Math.min(24 * 60 * 60 * 1000, 5 * 60 * 1000 * 2 ** Math.max(0, attemptCount - 1));
  }

  // --- Mapping helpers ---

  private mapPost(row: Record<string, unknown>): StoredPost {
    return {
      id: row.id as string,
      telegramId: Number(row.telegram_id),
      channelId: row.channel_id as string,
      text: row.text as string,
      urls: row.urls as string[],
      date: Number(row.date),
      processedAt: Number(row.processed_at),
    };
  }

  private mapArticle(row: Record<string, unknown>): StoredArticle {
    return {
      id: row.id as string,
      postId: (row.post_id as string) || undefined,
      sourceId: (row.source_id as string) || undefined,
      externalId: (row.external_id as string) || undefined,
      url: row.url as string,
      title: row.title as string,
      content: row.content as string,
      summary: (row.summary as string) || undefined,
      wordCount: Number(row.word_count),
      publishedAt: row.published_at ? Number(row.published_at) : undefined,
      processedAt: Number(row.processed_at),
    };
  }

  private mapSource(row: Record<string, unknown>): StoredSource {
    return {
      id: row.id as string,
      name: row.name as string,
      url: row.url as string,
      adapterType: row.adapter_type as string,
      adapterConfig: row.adapter_config as Record<string, unknown>,
      interestIds: row.interest_ids as string[],
      lastIngestedAt: row.last_ingested_at ? Number(row.last_ingested_at) : undefined,
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    };
  }

  private mapInterest(row: Record<string, unknown>): StoredInterest {
    return {
      id: row.id as string,
      slug: row.slug as string,
      name: row.name as string,
      description: row.description as string,
      embedding: parseVector(row.embedding, `interest embedding ${row.id as string}`),
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    };
  }

  private mapArticleExtractionJob(row: Record<string, unknown>): ArticleExtractionJob {
    return {
      url: row.url as string,
      postId: row.post_id as string,
      status: row.status as ArticleExtractionJob["status"],
      attemptCount: Number(row.attempt_count),
      lastError: (row.last_error as string) || undefined,
      nextAttemptAt: Number(row.next_attempt_at),
      lastAttemptAt: row.last_attempt_at ? Number(row.last_attempt_at) : undefined,
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    };
  }

  private mapEmbedding(row: Record<string, unknown>): StoredEmbedding {
    return {
      id: row.id as string,
      articleId: row.article_id as string,
      embedding: parseVector(row.embedding, `embedding ${row.id as string}`),
      model: row.model as string,
      createdAt: Number(row.created_at),
    };
  }

  private mapTopic(row: Record<string, unknown>): StoredTopic {
    return {
      id: row.id as string,
      name: row.name as string,
      description: row.description as string,
      centroidEmbedding: parseVector(row.centroid_embedding, `topic centroid ${row.id as string}`),
      articleCount: row.article_count as number,
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    };
  }

  private mapContentPlanItem(row: Record<string, unknown>): ContentPlanItem {
    return {
      id: row.id as string,
      topicId: row.topic_id as string,
      status: row.status as ContentPlanItem["status"],
      priority: row.priority as number,
      humanComment: (row.human_comment as string) || undefined,
      scheduledDate: row.scheduled_date ? Number(row.scheduled_date) : undefined,
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    };
  }

  private mapDraftArticle(row: Record<string, unknown>): DraftArticle {
    return {
      id: row.id as string,
      contentPlanId: row.content_plan_id as string,
      topicId: row.topic_id as string,
      title: row.title as string,
      body: row.body as string,
      tags: row.tags as string[],
      model: row.model as string,
      status: row.status as DraftArticle["status"],
      createdAt: Number(row.created_at),
      updatedAt: Number(row.updated_at),
    };
  }
}
