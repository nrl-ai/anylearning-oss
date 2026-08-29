import React from "react";
import { Swiper, SwiperSlide } from "swiper/react";
import { Navigation, Pagination, Autoplay } from "swiper/modules";
import Image from "next/image";
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/pagination";
import "swiper/css/autoplay";

export function Slideshow() {
  return (
    <div className="max-w-6xl mx-auto">
      <Swiper
        modules={[Navigation, Pagination, Autoplay]}
        spaceBetween={30}
        slidesPerView={1}
        navigation
        pagination={{ clickable: true }}
        autoplay={{
          delay: 3000,
          disableOnInteraction: false,
        }}
        loop={true}
        className="rounded-lg"
      >
        {[1, 2, 3, 4, 5].map((id) => (
          <SwiperSlide key={id}>
            <div className="px-4">
              <Image
                src={`/screenshots/${id}.png`}
                alt={`Screenshot ${id}`}
                width={1920}
                height={1080}
                className="rounded-lg"
              />
            </div>
          </SwiperSlide>
        ))}
      </Swiper>
    </div>
  );
}
