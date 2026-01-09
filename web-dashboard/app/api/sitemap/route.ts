import { NextResponse } from 'next/server';
import { SitemapGenerator, defaultSitemapConfig } from '@/lib/sitemap';

export async function GET() {
  try {
    const generator = new SitemapGenerator(defaultSitemapConfig);
    
    // Generate static pages
    const staticUrls = generator.generateStaticSitemap();
    
    // Generate dynamic pages
    const dynamicUrls = await generator.generateDynamicSitemap();
    
    // Combine all URLs
    const allUrls = [...staticUrls, ...dynamicUrls];
    
    // Generate XML
    const sitemapXML = generator.generateXML(allUrls);
    
    return new NextResponse(sitemapXML, {
      status: 200,
      headers: {
        'Content-Type': 'application/xml',
        'Cache-Control': 'public, max-age=3600, s-maxage=3600', // Cache for 1 hour
      },
    });
  } catch (error) {
    console.error('Error generating sitemap:', error);
    return new NextResponse('Error generating sitemap', { status: 500 });
  }
}

export const dynamic = 'force-dynamic';