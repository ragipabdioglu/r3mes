import { NextResponse } from 'next/server';
import { RSSGenerator, defaultRSSConfig, RSSItem } from '@/lib/sitemap';

export async function GET() {
  try {
    const generator = new RSSGenerator(defaultRSSConfig);
    
    // Generate RSS items (example data - replace with actual data source)
    const items: RSSItem[] = [
      {
        title: 'R3MES Network Launch',
        description: 'The R3MES decentralized AI mining network is now live! Start mining REMES tokens today.',
        link: 'https://r3mes.com/blog/network-launch',
        pubDate: new Date('2024-01-15').toUTCString(),
        guid: 'https://r3mes.com/blog/network-launch',
        author: 'team@r3mes.com (R3MES Team)',
        category: ['Announcement', 'Launch'],
      },
      {
        title: 'Mining Dashboard Updates',
        description: 'New features added to the R3MES mining dashboard including real-time statistics and improved UI.',
        link: 'https://r3mes.com/blog/dashboard-updates',
        pubDate: new Date('2024-01-10').toUTCString(),
        guid: 'https://r3mes.com/blog/dashboard-updates',
        author: 'dev@r3mes.com (R3MES Dev Team)',
        category: ['Updates', 'Dashboard'],
      },
      {
        title: 'AI Mining Algorithm Improvements',
        description: 'Enhanced AI algorithms for better mining efficiency and reduced energy consumption.',
        link: 'https://r3mes.com/blog/ai-improvements',
        pubDate: new Date('2024-01-05').toUTCString(),
        guid: 'https://r3mes.com/blog/ai-improvements',
        author: 'ai@r3mes.com (R3MES AI Team)',
        category: ['Technology', 'AI'],
      },
    ];
    
    // In a real application, you would fetch items from a database or CMS
    // const items = await getBlogPosts();
    
    const rssXML = generator.generateXML(items);
    
    return new NextResponse(rssXML, {
      status: 200,
      headers: {
        'Content-Type': 'application/rss+xml',
        'Cache-Control': 'public, max-age=3600, s-maxage=3600', // Cache for 1 hour
      },
    });
  } catch (error) {
    console.error('Error generating RSS feed:', error);
    return new NextResponse('Error generating RSS feed', { status: 500 });
  }
}

export const dynamic = 'force-dynamic';