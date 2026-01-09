// Sitemap generation utilities

export interface SitemapUrl {
  loc: string;
  lastmod?: string;
  changefreq?: 'always' | 'hourly' | 'daily' | 'weekly' | 'monthly' | 'yearly' | 'never';
  priority?: number;
}

export interface SitemapConfig {
  baseUrl: string;
  defaultChangefreq: SitemapUrl['changefreq'];
  defaultPriority: number;
}

export class SitemapGenerator {
  private config: SitemapConfig;

  constructor(config: SitemapConfig) {
    this.config = config;
  }

  generateXML(urls: SitemapUrl[]): string {
    const urlElements = urls.map(url => this.generateUrlElement(url)).join('\n');
    
    return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urlElements}
</urlset>`;
  }

  private generateUrlElement(url: SitemapUrl): string {
    const loc = url.loc.startsWith('http') ? url.loc : `${this.config.baseUrl}${url.loc}`;
    const lastmod = url.lastmod || new Date().toISOString().split('T')[0];
    const changefreq = url.changefreq || this.config.defaultChangefreq;
    const priority = url.priority !== undefined ? url.priority : this.config.defaultPriority;

    return `  <url>
    <loc>${loc}</loc>
    <lastmod>${lastmod}</lastmod>
    <changefreq>${changefreq}</changefreq>
    <priority>${priority}</priority>
  </url>`;
  }

  // Generate static pages sitemap
  generateStaticSitemap(): SitemapUrl[] {
    return [
      {
        loc: '/',
        changefreq: 'daily',
        priority: 1.0,
      },
      {
        loc: '/dashboard',
        changefreq: 'hourly',
        priority: 0.9,
      },
      {
        loc: '/mine',
        changefreq: 'daily',
        priority: 0.8,
      },
      {
        loc: '/network',
        changefreq: 'hourly',
        priority: 0.7,
      },
      {
        loc: '/chat',
        changefreq: 'daily',
        priority: 0.6,
      },
      {
        loc: '/docs',
        changefreq: 'weekly',
        priority: 0.5,
      },
      {
        loc: '/docs/getting-started',
        changefreq: 'weekly',
        priority: 0.5,
      },
      {
        loc: '/docs/mining-guide',
        changefreq: 'weekly',
        priority: 0.5,
      },
      {
        loc: '/docs/api',
        changefreq: 'weekly',
        priority: 0.4,
      },
      {
        loc: '/about',
        changefreq: 'monthly',
        priority: 0.3,
      },
      {
        loc: '/privacy',
        changefreq: 'yearly',
        priority: 0.2,
      },
      {
        loc: '/terms',
        changefreq: 'yearly',
        priority: 0.2,
      },
    ];
  }

  // Generate dynamic pages sitemap (e.g., blog posts, user profiles)
  async generateDynamicSitemap(): Promise<SitemapUrl[]> {
    const urls: SitemapUrl[] = [];

    try {
      // Example: Add blog posts
      // const posts = await getBlogPosts();
      // posts.forEach(post => {
      //   urls.push({
      //     loc: `/blog/${post.slug}`,
      //     lastmod: post.updatedAt,
      //     changefreq: 'monthly',
      //     priority: 0.6,
      //   });
      // });

      // Example: Add user profiles (if public)
      // const publicUsers = await getPublicUsers();
      // publicUsers.forEach(user => {
      //   urls.push({
      //     loc: `/user/${user.id}`,
      //     lastmod: user.updatedAt,
      //     changefreq: 'weekly',
      //     priority: 0.4,
      //   });
      // });

      return urls;
    } catch (error) {
      console.error('Error generating dynamic sitemap:', error);
      return [];
    }
  }

  // Generate robots.txt content
  generateRobotsTxt(): string {
    return `User-agent: *
Allow: /

# Disallow admin and private areas
Disallow: /admin/
Disallow: /api/
Disallow: /_next/
Disallow: /private/

# Allow specific API endpoints for SEO
Allow: /api/sitemap
Allow: /api/rss

# Sitemap location
Sitemap: ${this.config.baseUrl}/sitemap.xml

# Crawl delay (optional)
Crawl-delay: 1`;
  }
}

// RSS feed generation
export interface RSSItem {
  title: string;
  description: string;
  link: string;
  pubDate: string;
  guid?: string;
  author?: string;
  category?: string[];
}

export interface RSSConfig {
  title: string;
  description: string;
  link: string;
  language: string;
  managingEditor?: string;
  webMaster?: string;
  category?: string;
  ttl?: number;
}

export class RSSGenerator {
  private config: RSSConfig;

  constructor(config: RSSConfig) {
    this.config = config;
  }

  generateXML(items: RSSItem[]): string {
    const itemElements = items.map(item => this.generateItemElement(item)).join('\n');
    
    return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${this.escapeXML(this.config.title)}</title>
    <description>${this.escapeXML(this.config.description)}</description>
    <link>${this.config.link}</link>
    <language>${this.config.language}</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    <atom:link href="${this.config.link}/rss.xml" rel="self" type="application/rss+xml" />
    ${this.config.managingEditor ? `<managingEditor>${this.config.managingEditor}</managingEditor>` : ''}
    ${this.config.webMaster ? `<webMaster>${this.config.webMaster}</webMaster>` : ''}
    ${this.config.category ? `<category>${this.escapeXML(this.config.category)}</category>` : ''}
    ${this.config.ttl ? `<ttl>${this.config.ttl}</ttl>` : ''}
${itemElements}
  </channel>
</rss>`;
  }

  private generateItemElement(item: RSSItem): string {
    const categories = item.category?.map(cat => 
      `      <category>${this.escapeXML(cat)}</category>`
    ).join('\n') || '';

    return `    <item>
      <title>${this.escapeXML(item.title)}</title>
      <description>${this.escapeXML(item.description)}</description>
      <link>${item.link}</link>
      <pubDate>${item.pubDate}</pubDate>
      <guid isPermaLink="true">${item.guid || item.link}</guid>
      ${item.author ? `<author>${this.escapeXML(item.author)}</author>` : ''}
${categories}
    </item>`;
  }

  private escapeXML(str: string): string {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
}

// Default configurations
export const defaultSitemapConfig: SitemapConfig = {
  baseUrl: 'https://r3mes.com',
  defaultChangefreq: 'weekly',
  defaultPriority: 0.5,
};

export const defaultRSSConfig: RSSConfig = {
  title: 'R3MES Network Updates',
  description: 'Latest updates from the R3MES decentralized AI mining network',
  link: 'https://r3mes.com',
  language: 'en-us',
  managingEditor: 'team@r3mes.com (R3MES Team)',
  webMaster: 'webmaster@r3mes.com (R3MES Webmaster)',
  category: 'Technology',
  ttl: 60, // 1 hour
};