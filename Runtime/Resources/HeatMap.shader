Shader "NativeDijkstraMap/HeatMap"
{
    Properties
    {
        [NoScaleOffset] _MainTex ("Texture", 2D) = "white" {}
        MinCost ("Min Cost", Float) = 0
        MaxCost ("Max Cost", Float) = 10
        Alpha ("Alpha", Float) = 1
        FlowSpeed ("Flow Speed", Float) = 0
        FlowCycle ("Flow Cycle", Float) = 1
        [Enum(Inferno,0, Magma,1, Plasma,2, Viridis,3)] Gradient ("Gradient", Float) = 0
        FreeColor ("Free Color", Color) = (0, 0, 0, 0)
        ObstacleColor ("Obstacle Color", Color) = (.1, .1, .1, 1)
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Off ZWrite On ZTest Less

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #define FLT_MAX  3.4028235e+38
            #define FLT_MIN -3.4028235e+38

            #include "Gradient.hlsl"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            UNITY_DECLARE_TEX2D(_MainTex);
            float4 _MainTex_ST;
            float4 _MainTex_TexelSize;
            float MinCost;
            float MaxCost;
            float Alpha;
            float FlowSpeed;
            float FlowCycle;
            float Gradient;
            float4 FreeColor;
            float4 ObstacleColor;

            bool IsValid(float v, float2 coord)
            {
                float min = -100;
                float max = 100;

                if (v <= min)
                    return false;
                if (v >= max)
                    return false;

                if (coord.x < 0 || coord.y < 0 || coord.x >= 1 || coord.y >= 1)
                    return false;

                return true;
            }

            float SampleBilinear(v2f i)
            {
                // Do not blend pixels with sentinel values
                float pixel = UNITY_SAMPLE_TEX2D_LOD(_MainTex, i.uv, float2(0, 0));
                if (pixel <= FLT_MIN) return FLT_MIN;
                if (pixel >= FLT_MAX) return FLT_MAX;

                float2 textureSize   = _MainTex_TexelSize.zw; // (w, h)
                float2 texelSize = _MainTex_TexelSize.xy; // (1/w, 1/h)

                float2 st = i.uv * textureSize + 0.5;
                float2 i0 = floor(st);
                float2 f  = frac(st);

                float w00 = (1 - f.x) * (1 - f.y);
                float w10 = f.x       * (1 - f.y);
                float w01 = (1 - f.x) * f.y;
                float w11 = f.x       * f.y;

                float2 offset = - float2(0.999, 0.999);

                float2 uv00 = (i0 + float2(0, 0) + offset) * texelSize; // bottom-left center
                float2 uv10 = (i0 + float2(1, 0) + offset) * texelSize; // bottom-right center
                float2 uv01 = (i0 + float2(0, 1) + offset) * texelSize; // top-left center
                float2 uv11 = (i0 + float2(1, 1) + offset) * texelSize; // top-right center

                // Point-sample the four texels at a fixed mip
                float c00 = UNITY_SAMPLE_TEX2D_LOD(_MainTex, uv00, float2(0, 0));
                float c10 = UNITY_SAMPLE_TEX2D_LOD(_MainTex, uv10, float2(0, 0));
                float c01 = UNITY_SAMPLE_TEX2D_LOD(_MainTex, uv01, float2(0, 0));
                float c11 = UNITY_SAMPLE_TEX2D_LOD(_MainTex, uv11, float2(0, 0));

                // Mask out invalid values
                float sum = 0;
                float result = 0;
                bool anything = false;

                if (IsValid(c00, uv00)) { result += w00 * c00; sum += w00; anything = true;}
                if (IsValid(c10, uv10)) { result += w10 * c10; sum += w10; anything = true;}
                if (IsValid(c01, uv01)) { result += w01 * c01; sum += w01; anything = true;}
                if (IsValid(c11, uv11)) { result += w11 * c11; sum += w11; anything = true;}

                // If no valid samples, return sentinel
                if (!anything) return FLT_MAX;

                // Renormalize
                return result / sum;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // TODO: grid size parameter
                float2 st_grid = i.uv * _MainTex_TexelSize.zw;
                float2 f_grid  = frac(st_grid);
                float edge = 0.01;
                if (f_grid.x < edge || f_grid.y < edge || f_grid.x > 1-edge || f_grid.y > 1-edge) return FLT_MAX;

                // TODO: Option for point sampling vs bilinear
                //float value = UNITY_SAMPLE_TEX2D( _MainTex, i.uv );
                float value = SampleBilinear(i);

                if (value == 0) return float4(gradient(Gradient, 1), Alpha);  // Target
                if (value <= FLT_MIN) return FreeColor; // Free
                if (value >= FLT_MAX) return ObstacleColor; // Obstacle
                if (value > MaxCost) return float4(0, 0, 0, 0); // Too far away from target

                value = (value - MaxCost) / (MinCost - MaxCost);    // Scale
                value = FlowCycle*value - _Time.y*FlowSpeed;    // Animate Flow

                if (value > 1) value = value % 1;  // Modulo
                if (value < 0) value -= (int)value - 1;  // Modulo

                float normalized = saturate(value); // Normalize
                return float4(gradient(Gradient, normalized), Alpha);  // Gradient
            }
            ENDCG
        }
    }
}