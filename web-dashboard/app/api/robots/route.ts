import { NextResponse } from 'next/server';
import { SitemapGenerator, defaultSitemapConfig } from '@/lib/sitemap';

export async function GET() {
  try {
    const generator = new SitemapGenerator(defaultSitemapConfig);
    const robotsTxt = generator.generateRobotsTxt();
    
    return new NextResponse(robotsTxt, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain',
        'Cache-Control': 'public, max-age=86400, s-maxage=86400', // Cache for 24 hours
      },
    });
  } catch (error) {
    console.error('Error generating robots.txt:', error);
    return new NextResponse('Error generating robots.txt', { status: 500 });
  }
}

export const dynamic = 'force-dynamic';