import React, { useEffect, useRef } from "react";
import Hls from "hls.js";
import gsap from "gsap";
import { ArrowUpRight, Github, Linkedin, Twitter, Dribbble } from "lucide-react";

export default function Footer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const marqueeRef = useRef<HTMLDivElement>(null);
  const videoUrl = "https://stream.mux.com/Aa02T7oM1wH5Mk5EEVDYhbZ1ChcdhRsS2m1NYyx4Ua1g.m3u8";

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (Hls.isSupported()) {
      const hls = new Hls();
      hls.loadSource(videoUrl);
      hls.attachMedia(video);
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = videoUrl;
    }

    // Marquee Animation
    if (marqueeRef.current) {
      gsap.to(marqueeRef.current, {
        xPercent: -50,
        duration: 40,
        ease: "none",
        repeat: -1,
      });
    }
  }, []);

  return (
    <footer className="relative bg-bg pt-16 md:pt-32 pb-8 md:pb-12 overflow-hidden">
      {/* Background Video (Flipped) */}
      <div className="absolute inset-0 z-0 opacity-40">
        <video
          ref={videoRef}
          autoPlay
          muted
          loop
          playsInline
          className="absolute top-1/2 left-1/2 min-w-full min-h-full object-cover -translate-x-1/2 -translate-y-1/2 scale-y-[-1]"
        />
        <div className="absolute inset-0 bg-black/60" />
      </div>

      <div className="relative z-10">
        {/* Marquee */}
        <div className="border-y border-stroke py-8 mb-24 overflow-hidden whitespace-nowrap">
          <div ref={marqueeRef} className="inline-block text-4xl md:text-7xl lg:text-8xl font-display italic text-text-primary/20 uppercase tracking-tighter">
            {Array(10).fill("IDEA TO IMPACT • ").join("")}
          </div>
        </div>

        {/* CTA Content */}
        <div className="max-w-[1200px] mx-auto px-6 md:px-10 lg:px-16 text-center mb-32">
          <h2 className="text-5xl md:text-8xl lg:text-9xl font-display italic text-text-primary mb-12">
            Let's create <span className="text-muted">together</span>
          </h2>

          <a
            href="mailto:mustafamoizbohra@gmail.com"
            className="group relative inline-flex items-center gap-3 rounded-full px-10 py-5 text-lg text-text-primary transition-all hover:scale-105"
          >
            <div className="absolute inset-0 rounded-full accent-gradient opacity-0 group-hover:opacity-100 -m-[1px] transition-opacity" />
            <div className="relative z-10 flex items-center gap-3 bg-bg rounded-full px-10 py-5 -mx-10 -my-5">
              hello@mustafabohra.com <ArrowUpRight className="w-6 h-6" />
            </div>
          </a>
        </div>

        {/* Footer Bar */}
        <div className="max-w-[1200px] mx-auto px-6 md:px-10 lg:px-16 flex flex-col md:flex-row items-center justify-between gap-8 pt-12 border-t border-stroke">
          <div className="flex items-center gap-6">
            <SocialLink href="#" icon={<Twitter className="w-4 h-4" />} />
            <SocialLink href="#" icon={<Linkedin className="w-4 h-4" />} />
            <SocialLink href="#" icon={<Dribbble className="w-4 h-4" />} />
            <SocialLink href="#" icon={<Github className="w-4 h-4" />} />
          </div>

          <div className="text-[10px] text-muted uppercase tracking-[0.2em]">
            © 2026 Mustafa Bohra • All Rights Reserved
          </div>
        </div>
      </div>
    </footer>
  );
}

function SocialLink({ href, icon }: { href: string; icon: React.ReactNode }) {
  return (
    <a
      href={href}
      className="w-10 h-10 rounded-full border border-stroke flex items-center justify-center text-muted hover:text-text-primary hover:border-text-primary transition-all duration-300"
    >
      {icon}
    </a>
  );
}
