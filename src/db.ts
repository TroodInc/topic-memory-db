import pg from "pg";
import pgvector from "pgvector/pg";
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
  if (Array.isArray(value)) {
    return assertFiniteVector(value.map((item) => Number(item)), label);
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
      throw new Error(`${label} has invalid vector format`);
    }

    const parsed = trimmed
      .slice(1, -1)
      .split(",")
      .map((item) => Number(item.trim()));
    return assertFiniteVector(parsed, label);
  }

  throw new Error(`${label} has unsupported vector type`);
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
  CREATE EXTENSION IF NOT EXISTS vector;
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

  CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID NOT NULL REFERENCES posts(id),
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    word_count INTEGER NOT NULL,
    processed_at BIGINT NOT NULL
  );

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
    embedding vector(${EMBEDDING_DIM}) NOT NULL,
    model TEXT NOT NULL,
    created_at BIGINT NOT NULL
  );

  CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 20);

  CREATE TABLE IF NOT EXISTS topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    centroid_embedding vector(${EMBEDDING_DIM}) NOT NULL,
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
    this.pool = new pg.Pool({ connectionString });
  }

  /** Initialize schema and register pgvector type. Call once before use. */
  async init(): Promise<void> {
    if (this.initialized) return;
    const client = await this.pool.connect();
    try {
      await client.query(EXTENSIONS);
      await pgvector.registerTypes(client);
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

  /** Insert an article. Returns null if URL already exists. */
  async insertArticle(article: Omit<StoredArticle, "id" | "processedAt">): Promise<StoredArticle | null> {
    const id = randomUUID();
    const now = Date.now();
    try {
      await this.pool.query(
        `INSERT INTO articles (id, post_id, url, title, content, summary, word_count, processed_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
        [id, article.postId, article.url, article.title, article.content, article.summary || null, article.wordCount, now]
      );
      return { id, ...article, processedAt: now };
    } catch (e: unknown) {
      if (e instanceof Error && e.message.includes("duplicate key")) return null;
      throw e;
    }
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
    const vecLiteral = pgvector.toSql(assertFiniteVector(embedding.embedding, "embedding"));
    await this.pool.query(
      `INSERT INTO embeddings (id, article_id, embedding, model, created_at)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (article_id) DO UPDATE SET
         embedding = EXCLUDED.embedding,
         model = EXCLUDED.model,
         created_at = EXCLUDED.created_at`,
      [id, embedding.articleId, vecLiteral, embedding.model, now]
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

  /**
   * Find the K nearest embeddings to a query vector using pgvector.
   * Uses cosine distance operator (<=>).
   */
  async findNearestEmbeddings(
    queryEmbedding: number[],
    k = 5
  ): Promise<Array<StoredEmbedding & { distance: number }>> {
    const vecLiteral = pgvector.toSql(assertFiniteVector(queryEmbedding, "queryEmbedding"));
    const { rows } = await this.pool.query(
      `SELECT *, embedding <=> $1 AS distance
       FROM embeddings
       ORDER BY embedding <=> $1
       LIMIT $2`,
      [vecLiteral, k]
    );
    return rows.map((r: Record<string, unknown>) => ({
      ...this.mapEmbedding(r),
      distance: r.distance as number,
    }));
  }

  // --- Topics ---

  /** Insert or update a topic */
  async upsertTopic(topic: Omit<StoredTopic, "createdAt" | "updatedAt">): Promise<StoredTopic> {
    const now = Date.now();
    const vecLiteral = pgvector.toSql(
      assertFiniteVector(topic.centroidEmbedding, `topic centroid for ${topic.name}`)
    );
    await this.pool.query(
      `INSERT INTO topics (id, name, description, centroid_embedding, article_count, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (id) DO UPDATE SET
         name = EXCLUDED.name,
         description = EXCLUDED.description,
         centroid_embedding = EXCLUDED.centroid_embedding,
         article_count = EXCLUDED.article_count,
         updated_at = EXCLUDED.updated_at`,
      [topic.id, topic.name, topic.description, vecLiteral, topic.articleCount, now, now]
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
      postId: row.post_id as string,
      url: row.url as string,
      title: row.title as string,
      content: row.content as string,
      summary: (row.summary as string) || undefined,
      wordCount: row.word_count as number,
      processedAt: Number(row.processed_at),
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
