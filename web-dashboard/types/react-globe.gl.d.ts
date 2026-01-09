declare module 'react-globe.gl' {
  import { ComponentType } from 'react';

  interface GlobeProps {
    globeImageUrl?: string;
    pointsData?: Array<{
      lat: number;
      lng: number;
      size?: number;
      color?: string;
    }>;
    pointColor?: string | ((point: any) => string);
    pointRadius?: string | number | ((point: any) => number);
    pointResolution?: number;
    onGlobeReady?: () => void;
    [key: string]: any;
  }

  const Globe: ComponentType<GlobeProps>;
  export default Globe;
}