/** A Telegram post stored in the database */
export interface StoredPost {
  id: string;
  telegramId: number;
  channelId: string;
  text: string;
  urls: string[];
  date: number;
  processedAt: number;
}

/** An extracted article stored in the database */
export interface StoredArticle {
  id: string;
  postId: string;
  url: string;
  title: string;
  content: string;
  summary?: string;
  wordCount: number;
  processedAt: number;
}

/** An embedding vector stored in the database */
export interface StoredEmbedding {
  id: string;
  articleId: string;
  embedding: number[];
  model: string;
  createdAt: number;
}

/** A discovered topic */
export interface StoredTopic {
  id: string;
  name: string;
  description: string;
  centroidEmbedding: number[];
  articleCount: number;
  createdAt: number;
  updatedAt: number;
}

/** Link between a topic and an article */
export interface TopicArticleLink {
  topicId: string;
  articleId: string;
  similarity: number;
}

/** A generated article draft stored before publication */
export interface DraftArticle {
  id: string;
  contentPlanId: string;
  topicId: string;
  title: string;
  body: string;
  tags: string[];
  model: string;
  status: "draft" | "ready" | "published";
  createdAt: number;
  updatedAt: number;
}

/** A content plan entry */
export interface ContentPlanItem {
  id: string;
  topicId: string;
  status: "draft" | "approved" | "writing" | "ready" | "published" | "skipped";
  priority: number;
  humanComment?: string;
  scheduledDate?: number;
  createdAt: number;
  updatedAt: number;
}

/** A published article record */
export interface PublishedArticle {
  id: string;
  contentPlanId: string;
  discourseTopicId: number;
  discoursePostId: number;
  title: string;
  publishedAt: number;
}

/** Pipeline state for incremental processing */
export interface PipelineState {
  key: string;
  value: string;
  updatedAt: number;
}
